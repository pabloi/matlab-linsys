function [X,P,Xp,Pp,rejSamples]=statKalmanFilterFast(Y,A,C,Q,R,x0,P0,B,D,U,outlierRejection,fastFlag)
%filterStationary implements a Kalman filter assuming
%stationary (fixed) noise matrices and system dynamics
%The model is: x[k+1]=A*x[k]+b+v[k], v~N(0,Q)
%y[k]=C*x[k]+d+w[k], w~N(0,R)
%And X[0] ~ N(x0,P0) -> Notice that this is different from other
%implementations, where P0 is taken to be cov(x[0|-1]) so x[0]~N(x0,A*P0*A'+Q)
%See for example Ghahramani and Hinton 1996
%Fast implementation by assuming that filter's steady-state is reached after 20 steps

[D2,N]=size(Y);
%Init missing params:
if nargin<6 || isempty(x0)
  x0=zeros(size(A,1),1); %Column vector
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
elseif fastFlag==0
    M2=20; %Default for fast filtering: 20 samples
    M1=ceil(3*max(-1./log(abs(eig(A))))); %This many strides ensures ~convergence of gains before we assume steady-state
    M=max(M1,M2);
    M=min(M,N); %Prevent more than N, if this happens, we are not doing fast filtering
else
    M=min(ceil(abs(fastFlag)),N); %If fastFlag is a number but not 0, use that as number of samples
end

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
    Xp=nan(size(A,1),N+1,'gpuArray');
    X=nan(size(A,1),N,'gpuArray');
    Pp=nan(size(A,1),size(A,1),N+1,'gpuArray');
    P=nan(size(A,1),size(A,1),N,'gpuArray');
    rejSamples=zeros(D2,N,'gpuArray');
else
    Xp=nan(size(A,1),N+1);
    X=nan(size(A,1),N);
    Pp=nan(size(A,1),size(A,1),N+1);
    P=nan(size(A,1),size(A,1),N);
    rejSamples=zeros(D2,N);
end

%Priors:
tol=1e-8;
prevX=x0;
prevP=P0;
Xp(:,1)=x0;
Pp(:,:,1)=P0;

%Pre-computing for speed:
CtRinv=C'*pinv(R,tol); %gpu-ready  %Equivalent to lsqminnorm(R,C,tol)';, not gpu ready
CtRinvC=CtRinv*C;

Y_D=Y-D*U;
CtRinvY=CtRinv*Y_D;
BU=B*U;

%Do the true filtering for M steps
for i=1:M
  %First, do the update given the output at this step:
  if ~outlierRejection
      [prevX,prevP]=KFupdate(CtRinvY(:,i),CtRinvC,prevX,prevP);
  else
      warning('Outlier rejection not implemented')
      %TODO: reject outliers here
      %[outlierIndx]=detectOutliers(Y_D(:,i),x,P,C,R,rejectThreshold);
     [prevX,prevP]=KFupdate(CtRinvY(:,i),CtRinvC,prevX,prevP);
  end
  X(:,i)=prevX;
  P(:,:,i)=prevP;
  
  %Then, predict next step:
  [prevX,prevP]=KFpredict(A,Q,prevX,prevP,BU(:,i));
  Xp(:,i+1)=prevX;
  Pp(:,:,i+1)=prevP;
end

%Do the fast filtering for any remaining steps:
if M<N
    %Steady-state matrices:
    Psteady=prevP; %Steady-state predicted uncertainty matrix
    iPsteady=prevP\eye(size(prevP));%For some reason, this is much faster than pinv
    Ksteady=(iPsteady+CtRinvC)\iPsteady;
    innov=(iPsteady+CtRinvC)\CtRinvY;
    P(:,:,M+1:end)=repmat(P(:,:,M),1,1,size(Y,2)-M);
    %Pre-compute matrices to reduce computing time:
    KBUY=Ksteady*BU+innov;
    KA=Ksteady*A;
    %Loop for remaining steps:
    for i=M+1:size(Y,2)
        prevX=KA*prevX+KBUY(:,i); %Predict+Update
        X(:,i)=prevX;
    end
    %Compute Xp, Pp if requested:
    if nargout>2
        Xp(:,2:end)=A*X+B*U;
        Pp(:,:,M+2:end)=repmat(Psteady,1,1,size(Y,2)-M);
    end
end
end
