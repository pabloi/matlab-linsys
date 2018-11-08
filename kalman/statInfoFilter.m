function [X,P,Xp,Pp,rejSamples,logL,Ip]=statInfoFilter(Y,A,C,Q,R,varargin)
%statInfoFilter implements the Information formulation of the Kalman filter 
%assuming stationary (fixed) noise matrices and system dynamics
%The model is: x[k+1]=A*x[k]+b+v[k], v~N(0,Q)
%y[k]=C*x[k]+d+w[k], w~N(0,R)
%INPUTS:
%
%OUTPUTS:
%
%See also: statInfoSmoother, statKalmanFilter, infoUpdate, KFupdate


[D2,N]=size(Y); D1=size(A,1);
%Init missing params:
[x0,P0,B,D,~,Ud,Ub,opts]=processKalmanOpts(D1,N,varargin);
M=processFastFlag(opts.fastFlag,A,N);

%TODO: Special case: deterministic system, no filtering needed. This can also be
%the case if Q << C'*R*C, and the system is stable

%Size checks:
%TODO

%Init arrays:
if isa(Y,'gpuArray') %For code to work on gpu
    Xp=nan(D1,N+1,'gpuArray');      X=nan(D1,N,'gpuArray');
    Pp=nan(D1,D1,N+1,'gpuArray');   P=nan(D1,D1,N,'gpuArray');
    rejSamples=false(D2,N,'gpuArray'); Ip=nan(D1,D1,N+1,'gpuArray');
else
    Xp=nan(D1,N+1);      X=nan(D1,N);
    Pp=nan(D1,D1,N+1);   P=nan(D1,D1,N);
    Ip=nan(D1,D1,N+1);
    rejSamples=false(D2,N); 
end

%Priors:
prevX=x0; prevP=P0; Xp(:,1)=x0; Pp(:,:,1)=P0;

%Re-define observations to account for input effect:
Y_D=Y-D*Ud; BU=B*Ub;

%Define constants for sample rejection:
logL=nan(1,N); %Row vector
if opts.outlierFlag
    warning('Sample rejection not implemented for information filter')
end

%Precompute for efficiency:
[CtRinvC,~,CtRinvY,~,logLmargin]=reduceModel(C,R,Y_D); 

%For the first steps do an information update if P0 contains infinite elements
infVariances=isinf(diag(prevP));
if any(infVariances)
    %Define info matrix from cov matrix:
    prevI=zeros(size(prevP));
    aux=inv(prevP(~infVariances,~infVariances)); %This inverse needs to exist, no such thing as absolute certainty
    prevI(~infVariances,~infVariances)=aux; %Computing inverse of the finite submatrix of P0
else
    prevI=inv(prevP); %This inverse needs to exist, no such thing as absolute certainty
end

%Run filter:
if M<N %Do the fast filtering for any remaining steps:
    warning('Fast mode not implemented in information filter')
    M=N;
end
for i=1:M
  y=CtRinvY(:,i); %Output at this step

  %First, do the update given the output at this step:
  if ~any(isnan(y)) %If measurement is NaN, skip update.
     [~,~,prevX,prevP]=infoUpdate(CtRinvC,y,prevX,[],prevI);
  end
  X(:,i)=prevX;  P(:,:,i)=prevP; %Store results

  %Then, predict next step:
  [prevX,prevP]=KFpredict(A,Q,prevX,prevP,BU(:,i));
  [~,~,prevI]=pinvchol(prevP);
  if nargout>2 %Store Xp, Pp if requested:
      Xp(:,i+1)=prevX;   Pp(:,:,i+1)=prevP; 
      if nargout>5; Ip(:,:,i+1)=prevI; end
  end
end

%Compute mean log-L over samples and dimensions of the output:
logL=nanmean(logL+logLmargin)/size(Y,1);
end
