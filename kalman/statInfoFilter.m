function [X,P,Xp,Pp,rejSamples,logL,Ip,I]=statInfoFilter(Y,A,C,Q,R,varargin)
%statInfoFilter implements the Information formulation of the Kalman filter 
%assuming stationary (fixed) noise matrices and system dynamics
%The model is: x[k+1]=A*x[k]+b+v[k], v~N(0,Q)
%y[k]=C*x[k]+d+w[k], w~N(0,R)
%This is NOT a pure information filter implementation. Rather it is a
%hybrid that uses the information formulation for the measurement update,
%but the classic formulation for the update. This results in a more
%efficient filter. However, if the log-likelihood is to be computed, then
%this formulation is NOT faster than the classic Kalman one using a reduced
%model when dim(y)>dim(x). It is faster if log-L is not computed.
%INPUTS:
%
%OUTPUTS:
%
%See also: statInfoSmoother, statKalmanFilter, infoUpdate, KFupdate

%To do: use statInfoFilter2 (trueStatInfoFilter) and enforce conversion
%through state space (when priors are proper) so X,P are computed in the
%filtering stage itself, and there is no need to add a conversion stage at
%the end

error('Deprecated: use statInfoFilter2')

[D2,N]=size(Y); D1=size(A,1);
%Init missing params:
[x0,P0,B,D,U,opts]=processKalmanOpts(D1,N,varargin);
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
    Ip=nan(D1,D1,N+1);   I=nan(D1,D1,N);
    rejSamples=false(D2,N); 
end

%Priors:
prevX=x0; prevP=P0; Xp(:,1)=x0; Pp(:,:,1)=P0;

%Re-define observations to account for input effect:
Y_D=Y-D*U; BU=B*U;

%Define constants for sample rejection:
logL=nan(1,N); %Row vector
if opts.outlierFlag
    warning('Sample rejection not implemented for information filter')
end

%Precompute for efficiency:
[CtRinvC,~,CtRinvY,~,logLmargin]=reduceModel(C,R,Y_D); 
[cholInvCRC,~,invCRC]=pinvldl(CtRinvC);
logDetCRC=-2*sum(log(diag(cholInvCRC)));
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
     [~,thisI,prevX,prevP,logL(i),rejSamples(i),prevI]=infoUpdate(CtRinvC,y,prevX,prevP,[],logDetCRC,invCRC);
  end
  X(:,i)=prevX;  P(:,:,i)=prevP; %Store results
  
  %Then, predict next step:
  if i==1 && any(isinf(diag(prevP)))
      warning('Infinte covariance matrix at predict step, no promises this will be handled well. Try using statInfoFilter2 or large, but finite, variances.')
  end
  [prevX,prevP]=KFpredict(A,Q,prevX,prevP,BU(:,i));
  if nargout>2 %Store Xp, Pp if requested:
      Xp(:,i+1)=prevX;   Pp(:,:,i+1)=prevP; 
      if nargout>5; Ip(:,:,i)=prevI; I(:,:,i)=thisI; end %Storing the value for the previous step
  end
end

%Compute mean log-L over samples and dimensions of the output:
logL=nanmean(logL+logLmargin)/size(Y,1);
end
