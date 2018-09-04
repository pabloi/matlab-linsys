function [X,P,Xp,Pp,rejSamples]=statKalmanFilterConstrained(Y,A,C,Q,R,x0,P0,B,D,U,outlierRejection,constrFun)
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


%Size checks:
%TODO

  Xp=nan(size(A,1),N+1);
  X=nan(size(A,1),N);
  Pp=nan(size(A,1),size(A,1),N+1);
  P=nan(size(A,1),size(A,1),N);
  rejSamples=zeros(D2,N);

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
for i=1:N
  %First, do the update given the output at this step:
  CiRy=CtRinvY(:,i);
  if outlierRejection
      %TODO: reject outliers by replacing with NaN
      warning('Outlier rejection not implemented')
  end
  if ~any(isnan(CiRy)) %If measurement is NaN, skip update.
      [prevX,prevP]=KFupdate(CiRy,CtRinvC,prevX,prevP);
  end
  %Enforce constraint:\
  [H,b]=constrFun(prevX); %Linearized constraints, such that in a neigborhood around prevX  the constraint can be approximated as: H*x=b
  prevX=prevX-pinv(H)*(H*prevX-b);
  cP=chol(prevP);
  I=eye(size(prevP));
  aux=(I-pinv(H)*H)*cP';
  prevP=aux*aux'+1e-12*I; %Numerical conditioning

  X(:,i)=prevX;
  P(:,:,i)=prevP;

  %Then, predict next step:
  [prevX,prevP]=KFpredict(A,Q,prevX,prevP,BU(:,i));
  Xp(:,i+1)=prevX;
  Pp(:,:,i+1)=prevP;
end

end
