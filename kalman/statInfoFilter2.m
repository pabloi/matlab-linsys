function [ii,I,ip,Ip]=statInfoFilter2(Y,A,C,Q,R,varargin)
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
    ip=nan(D1,N+1,'gpuArray');      ii=nan(D1,N,'gpuArray');
    Ip=nan(D1,D1,N+1,'gpuArray');   I=nan(D1,D1,N,'gpuArray');
else
    ip=nan(D1,N+1);      ii=nan(D1,N);
    Ip=nan(D1,D1,N+1);   I=nan(D1,D1,N);
end

%Priors:
prevP=P0;

%Re-define observations to account for input effect:
Y_D=Y-D*U; BU=B*U;

%Define constants for sample rejection:
logL=nan(1,N); %Row vector
if opts.outlierFlag
    warning('Sample rejection not implemented for information filter')
end

%Precompute for efficiency:
[CtRinvC,~,CtRinvY,~,logLmargin]=reduceModel(C,R,Y_D);
[cholInvCRC,~,invCRC]=pinvchol(CtRinvC);
logDetCRC=-2*sum(log(diag(cholInvCRC)));
[~,~,iQ]=pinvchol(Q);
iQA=iQ*A;
AtiQA=A'*iQ*A;

%Sanitize init Info matrix if P0 contains infinite elements
infVariances=isinf(diag(prevP));
if any(infVariances)
    %Define info matrix from cov matrix:
    prevI=zeros(size(prevP));
    aux=inv(prevP(~infVariances,~infVariances)); %This inverse needs to exist, no such thing as absolute certainty
    prevI(~infVariances,~infVariances)=aux; %Computing inverse of the finite submatrix of P0
else
    prevI=inv(prevP); %This inverse needs to exist, no such thing as absolute certainty
end
previ=prevI*x0;
ip(:,1)=previ;
Ip(:,:,1)=prevI;

%Run filter:
if M<N %Do the fast filtering for any remaining steps:
    warning('Fast mode not implemented in information filter')
    M=N;
end
for i=1:M
  y=CtRinvY(:,i); %Output at this step

  %First, do the update given the output at this step:
  if ~any(isnan(y)) %If measurement is NaN, skip update.
     %[~,thisI,prevX,prevP,logL(i),rejSamples(i),prevI]=infoUpdate(CtRinvC,y,prevX,prevP,[],logDetCRC,invCRC);
     [previ,prevI]=infoUpdate2(CtRinvC,y,previ,prevI);
  end
  ii(:,i)=previ;  I(:,:,i)=prevI; %Store results

  %Then, predict next step: (if prevI==0 there is no need for this)
  if opts.fastFlag
    auxP=iQA/(prevI+AtiQA);
    prevI=iQ-auxP*iQA'; %This is not guaranteed to be symmetric in this form
    previ=auxP*previ + prevI*BU(:,i);
  else
      [~,cholAuxP]=pinvchol(prevI+AtiQA);
      H=iQA*cholAuxP;
      prevI=iQ-H*H'; %This should be exactly 0 if prevI was 0
      previ=H*cholAuxP'*previ + prevI*BU(:,i);
  end

  %[prevX,prevP]=KFpredict(A,Q,prevX,prevP,BU(:,i));
  if nargout>2 %Store Xp, Pp if requested:
      ip(:,i+1)=previ;   Ip(:,:,i+1)=prevI;
  end
end
end
