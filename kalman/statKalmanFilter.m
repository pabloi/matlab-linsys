function [X,P,Xp,Pp,rejSamples]=statKalmanFilter(Y,A,C,Q,R,x0,P0,B,D,U,outlierRejection,fastFlag)
%filterStationary implements a Kalman filter assuming
%stationary (fixed) noise matrices and system dynamics
%The model is: x[k+1]=A*x[k]+b+v[k], v~N(0,Q)
%y[k]=C*x[k]+d+w[k], w~N(0,R)
%And X[0] ~ N(x0,P0) -> Notice that this is different from other
%implementations, where P0 is taken to be cov(x[0|-1]) so x[0]~N(x0,A*P0*A'+Q)
%See for example Ghahramani and Hinton 1996
%Fast implementation by assuming that filter's steady-state is reached after 20 steps
%INPUTS:
%
%fastFlag: flag to indicate if fast smoothing should be performed. Default is no. Empty flag means no, any other value is yes.
%OUTPUTS:
%
%See also: statKalmanSmoother, statKalmanFilterConstrained, KFupdate, KFpredict

%TODO: check relative size of R,P and use the more efficient kalman update
%for each case. Will need to check if P is invertible, which will generally
%be the case if Q is invertible. If not, probably need to do the
%conventional update too.

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


[D2,N]=size(Y);
D1=size(A,1);
%Init missing params:
if nargin<6 || isempty(x0)
  x0=zeros(D1,1); %Column vector
end
if nargin<7 || isempty(P0)
  P0=1e8 * eye(size(A));
end
if nargin<8 || isempty(B)
  B=0;
end
if nargin<9 || isempty(D)
  D=0;
end
if nargin<10 || isempty(U)
  U=zeros(size(B,2),size(X,2));
end
if nargin<11 || isempty(outlierRejection)
    outlierRejection=false;
end
if nargin<12 || isempty(fastFlag)
    M=N; %Do true filtering for all samples
elseif any(any(isnan(Y)))
  warning('statKFfast:NaNsamples','Requested fast KF but some samples are NaN, not using fast mode.')
  M=N;
elseif fastFlag==0
    M2=20; %Default for fast filtering: 20 samples
    M1=ceil(3*max(-1./log(abs(eig(A))))); %This many strides ensures ~convergence of gains before we assume steady-state
    M=max(M1,M2);
    M=min(M,N); %Prevent more than N, if this happens, we are not doing fast filtering
else
    M=min(ceil(abs(fastFlag)),N); %If fastFlag is a number but not 0, use that as number of samples
end

%Special case: deterministic system, no filtering needed. This can also be
%the case if Q << C'*R*C, and the system is stable
% if all(Q(:)==0)
%     [~,Xp]=fwdSim(U,A,B,C,D,x0,[],[]);
%     Ndim=size(x0,1);
%     Pp=zeros(Ndim,Ndim,N+1);
%     X=Xp(:,1:end-1);
%     P=zeros(Ndim,Ndim,N);
%     rejSamples=[];
%     return
% end

%Size checks:
%TODO

if M<N && any(abs(eig(A))>1)
    %If the system is unstable, there is no guarantee that the kalman gain
    %converges, and the fast filtering will lead to divergence of estimates
    warning('statKSfast:unstable','Doing steady-state (fast) filtering on an unstable system. States will diverge. Doing traditional filtering instead.')
    M=N;
end

%Init arrays:
if isa(Y,'gpuArray') %For code to work on gpu
    Xp=nan(D1,N+1,'gpuArray');
    X=nan(D1,N,'gpuArray');
    Pp=nan(D1,D1,N+1,'gpuArray');
    P=nan(D1,D1,N,'gpuArray');
    rejSamples=zeros(D2,N,'gpuArray');
else
    Xp=nan(D1,N+1);
    X=nan(D1,N);
    Pp=nan(D1,D1,N+1);
    P=nan(D1,D1,N);
    rejSamples=zeros(D2,N);
end

%Priors:
prevX=x0;
prevP=P0;
Xp(:,1)=x0;
Pp(:,:,1)=P0;

%Re-define observations to account for input effect:
Y_D=Y-D*U;
BU=B*U;

%If D2>D1, then it is speedier to do a coordinate transform of the output:
%(it may also be convenient to do something if C is not full rank,as that
%means the output is also compressible). This is always safe if R is
%invertible, and may be safe in other situations, provided that
%observations never fall on the null-space of R.
if D2>D1
    cR=chol(R);% mycholcov(R); could do if R is semidefinite, but in general semidefinite R is unworkable, as R+C*P*C' needs to be invertible. 
    %Even if we assume P invertible, that still requires R to be invertible for all vectors orthogonal to the span of C at least)
    J=C'/cR';
    icR=eye(size(R))/cR;
    Rinv=icR*icR'; 
    %tol=1e-8;
    %Rinv =pinv(R,tol); works for non-invertible R, although the results are not quite what we'd expect. 
    %If pinv() is used, we effectively eliminate the projection of Y_D onto
    %the nulll space of R, although that is not what the standard Kalman
    %filter would do. Perhaps the optimal behavior then would be to
    %decompose the update into two parts: the R null component, which has a
    %Kalman gain of K=PC'/(CPC') [which characterizes a well-defined filter
    %as long as the null space of R is contained within the span of C], and
    %the R-potent component, which has the standard Kalman gain, and where
    %Rinv is well-defined.
    %Redefine observations and obs equation to the dim reduced form:
    R=J*J';
    Y_D=C'*Rinv*Y_D;
    C=R;
end

%Do the true filtering for M steps
for i=1:M
  %First, do the update given the output at this step:
  y=Y_D(:,i);
  if ~any(isnan(y)) %If measurement is NaN, skip update.
      if outlierRejection; [prevXt,prevPt,z]=KFupdate(C,R,y,x,P);
          if z<th %zscore is less than the outlier threshold, doing update
              prevP=prevPt;    prevX=prevXt;
          end
      else %This is here because it is more efficient to not compute the z-score if we dont need it
          [prevX,prevP,K]=KFupdate(C,R,y,prevX,prevP);
      end
  end
  X(:,i)=prevX;  P(:,:,i)=prevP; %Store results

  %Then, predict next step:
  [prevX,prevP]=KFpredict(A,Q,prevX,prevP,BU(:,i));
  if nargout>2 %Store Xp, Pp if requested:
      Xp(:,i+1)=prevX;   Pp(:,:,i+1)=prevP;
  end
end

%Do the fast filtering for any remaining steps: 
%(from here on, we assume stady-state behavior to improve speed).
if M<N 
    %Steady-state matrices:
    Psteady=P(:,:,M); %Steady-state UPDATED uncertainty matrix
    prevX=X(:,M);
    Ksteady=K; %Steady-state Kalman gain
    Gsteady=eye(size(Ksteady,1))-Ksteady*C; %I-K*C,
    
    %Pre-compute matrices to reduce computing time:
    GBU_KY=Gsteady*BU(:,M:N-1)+Ksteady*Y_D(:,M+1:N); %The off-ordering is because we are doing predict (which depends on U(:,i-1)) and update (which depends on Y(:,i)
    GA=Gsteady*A;
    
    %Assign all UPDATED uncertainty matrices:
    P(:,:,M+1:end)=repmat(P(:,:,M),1,1,N-M);
    
    %Loop for remaining steps to compute x:
    if outlierRejection
        %TODO: reject outliers by replacing with NaN in KBUY, this needs to be done in-loop
       warning('Outlier rejection not implemented')
    end
    for i=M+1:N
        gbu_ky=GBU_KY(:,i-M);
        if ~any(isnan(gbu_ky))
            prevX=GA*prevX+gbu_ky; %Predict+Update, in that order.
            %TODO: evaluate if this is good: because we dont compute y-C*X first 
            %and then multiply by, K, we may be accumulating numerical errors in 
            %cases where (y-C*x)==0
        else %Reading is NaN, just update. Should never happen since data with NaNs prevents fast filtering.
            %prevX=A*prevX+BU(:,i); %Just predict
            error('Skipping update for a NaN sample, but uncertainty does not get updated accordingly in fast mode. FIX.')
        end
        X(:,i)=prevX;
    end
    if nargout>2 %Compute Xp, Pp only if requested:
        Xp(:,2:end)=A*X+B*U;
        Pp(:,:,M+2:end)=repmat(A*Psteady*A'+Q,1,1,size(Y,2)-M);
    end
end
end
