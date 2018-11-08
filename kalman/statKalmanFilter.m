function [X,P,Xp,Pp,rejSamples,logL,invSchol]=statKalmanFilter(Y,A,C,Q,R,varargin)
%statKalmanFilter implements a Kalman filter assuming
%stationary (fixed) noise matrices and system dynamics
%The model is: x[k+1]=A*x[k]+b+v[k], v~N(0,Q)
%y[k]=C*x[k]+d+w[k], w~N(0,R)
%And X[0] ~ N(x0,P0) -> Notice that this is different from other
%implementations, where P0 is taken to be cov(x[0|-1]) so x[0]~N(x0,A*P0*A'+Q)
%See for example Ghahramani and Hinton 1996
%Fast implementation by assuming that filter's steady-state is reached after 20 steps
%INPUTS:
%
%OUTPUTS:
%
%See also: statKalmanSmoother, statKalmanFilterConstrained, KFupdate, KFpredict

% For the filter to be well-defined, it is necessary that the quantity w=C'*inv(R+C*P*C')*y
% be well defined, for all observations y with some reasonable definition of inv().
% Naturally, this is the case if R+C*P*C' is invertible at each step. In
% turn, this is always the case if R is invertible, as P is positive semidef.
% There may exist situations where w is well defined even if R+C*P*C' is
% not invertible (which implies R is non inv). %This requires both: the
% projection of y and of the columns of C onto the 'uninvertible' subspace
% to be  always 0. In that case the output space can be 'compressed' to a
% smaller dimensional one by eliminating nuisance dimensions. This can be done because
% neither the state projects onto those dims, nor the observations fall in it.
% Such a reduction of the output space can be done for efficiency even if
% the projection of y is non-zero, provided that R is invertible and the
% structure of R decouples those dimensions from the rest (i.e. the
% observations along those dims are uncorrelated to the dims corresponding
% to the span of C). Naturally, this is a very special case, but there are
% some easy-to-test sufficient conditions: if R is diagonal, positive, and
% rank(C)<dim(R), compression is always possible.


[D2,N]=size(Y); D1=size(A,1);
%Init missing params:
[x0,P0,B,D,U,Ud,Ub,opts]=processKalmanOpts(D1,N,varargin);
M=processFastFlag(opts.fastFlag,A,N);

%TODO: Special case: deterministic system, no filtering needed. This can also be
%the case if Q << C'*R*C, and the system is stable

%Size checks:
%TODO

%Init arrays:
if isa(Y,'gpuArray') %For code to work on gpu
    Xp=nan(D1,N+1,'gpuArray');      X=nan(D1,N,'gpuArray');
    Pp=nan(D1,D1,N+1,'gpuArray');   P=nan(D1,D1,N,'gpuArray');
    rejSamples=false(D2,N,'gpuArray');
else
    Xp=nan(D1,N+1);      X=nan(D1,N);
    Pp=nan(D1,D1,N+1);   P=nan(D1,D1,N);
    rejSamples=false(D2,N); 
end

%Priors:
prevX=x0; prevP=P0; Xp(:,1)=x0; Pp(:,:,1)=P0;

%Re-define observations to account for input effect:
Y_D=Y-D*Ud; BU=B*Ub;

%Define constants for sample rejection:
logL=nan(1,N); %Row vector
rejThreshold=[];
if opts.outlierFlag
  rejThreshold=chi2inv(.99,D2);
end

%Reduce model if convenient for efficiency:
[CtRinvC,~,CtRinvY,cholCtRinvC,logLmargin]=reduceModel(C,R,Y_D); 
if D2>D1 && ~opts.noReduceFlag %Reducing dimension of problem for speed
    C=CtRinvC; R=CtRinvC; Y_D=CtRinvY;  D2=D1; cR=cholCtRinvC; rejSamples=rejSamples(1:D1,:);
else
    cR=mycholcov(R);  logLmargin=0;
end
if nargout>5
    invSchol=nan(D2,D2,N);
end

%For the first steps do an information update if P0 contains infinite elements
firstInd=1;
infVariances=isinf(diag(prevP));
while any(infVariances) %In practice, this only gets executed once at most.
    %Define info matrix from cov matrix:
    prevI=zeros(size(prevP));
    aux=inv(prevP(~infVariances,~infVariances)); %This inverse needs to exist, no such thing as absolute certainty
    prevI(~infVariances,~infVariances)=aux; %Computing inverse of the finite submatrix of P0
    %prevI=diag(1./diag(P0)); %This information matrix ignores correlations, cheaper
    %Update:
    [~,~,prevX,prevP,logL(firstInd)]=infoUpdate(CtRinvC,CtRinvY(:,1),prevX,prevP,prevI);
    X(:,firstInd)=prevX;  P(:,:,firstInd)=prevP; %Store results
    %Predict:
    [prevX,prevP]=KFpredict(A,Q,prevX,prevP,BU(:,1));
    firstInd=firstInd+1;
    Xp(:,firstInd)=prevX;   Pp(:,:,2)=prevP; 
    %New variances:
    infVariances=isinf(diag(prevP));
end

%Run filter for remaining steps:
for i=firstInd:M
  y=Y_D(:,i); %Output at this step

  %First, do the update given the output at this step:
  if ~any(isnan(y)) %If measurement is NaN, skip update.
     [prevX,prevP,prevK,logL(i),rejSamples(i),icS]=KFupdate(C,R,y,prevX,prevP,rejThreshold,cR);
  end
  X(:,i)=prevX;  P(:,:,i)=prevP; %Store results

  %Then, predict next step:
  [prevX,prevP]=KFpredict(A,Q,prevX,prevP,BU(:,i));
  if nargout>2 %Store Xp, Pp if requested:
      Xp(:,i+1)=prevX;   Pp(:,:,i+1)=prevP; 
      if nargout>5; invSchol(:,:,i)=icS; end
  end
end

if M<N %Do the fast filtering for any remaining steps:
%(from here on, we assume stady-state behavior to improve speed).
    %Steady-state matrices:
    prevX=X(:,M); Psteady=P(:,:,M); %Steady-state UPDATED state and uncertainty matrix
    Ksteady=prevK; %Steady-state Kalman gain
    Gsteady=eye(size(Ksteady,1))-Ksteady*C; %I-K*C,

    %Pre-compute matrices to reduce computing time:
    GBU_KY=Gsteady*BU(:,M:N-1)+Ksteady*Y_D(:,M+1:N); %The off-ordering is because we are doing predict (which depends on U(:,i-1)) and update (which depends on Y(:,i)
    GA=Gsteady*A;

    %Assign all UPDATED uncertainty matrices:
    P(:,:,M+1:end)=repmat(Psteady,1,1,N-M);

    %Check that no outlier or fast flags are enabled
    if opts.outlierFlag || any(isnan(GBU_KY(:)))%Should never happen in fast mode
       error('KFfilter:outlierRejectFast','Outlier rejection is incompatible with fast mode.')
    end

    %Loop for remaining steps to compute x:
    for i=M+1:N
        gbu_ky=GBU_KY(:,i-M);
        prevX=GA*prevX+gbu_ky; %Predict+Update, in that order.
        X(:,i)=prevX;
    end
    if nargout>2 %Compute Xp, Pp only if requested:
        Xp(:,2:end)=A*X+B*Ub; Pp(:,:,M+2:end)=repmat(A*Psteady*A'+Q,1,1,size(Y,2)-M);
        if nargout>4; Innov=Y_D-C*Xp(:,1:end-1);  logL(M+1:end)=logLnormal(Innov(:,M+1:end),[],icS');
            if nargout>5; invSchol(:,:,M+1:end)=repmat(icS,1,1,size(Y,2)-M); end
        end
    end
end

%Compute mean log-L over samples and dimensions of the output:
logL=nanmean(logL+logLmargin)/size(Y,1);
end
