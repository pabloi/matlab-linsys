function [X,Pinv,Xp,priorPinv,rejSamples]=statKalmanFilterEff(Y,A,C,Q,R,x0,P0,B,D,U,outlierRejection)
%filterStationary implements a Kalman filter assuming
%stationary (fixed) noise matrices and system dynamics
%The model is: x[k+1]=A*x[k]+b+v[k], v~N(0,Q)
%y[k]=C*x[k]+d+w[k], w~N(0,R)
%And X[0] ~ N(x0,P0) -> Notice that this is different from other
%implementations, where P0 is taken to be cov(x[0|-1]) so x[0]~N(x0,A*P0*A'+Q)
%See for example Ghahramani and Hinton 1996

tol=1e-8;
%Init missing params:
if nargin<6 || isempty(x0)
  x0=zeros(size(A,1),1); %Column vector
end
if nargin<7 || isempty(P0)
  P0inv=zeros(size(A));
else
    P0inv=pinv(P0,tol);
end
if nargin<8 || isempty(B)
  B=0;
end
if nargin<9 || isempty(D)
  D=0;
end
if nargin<10 || isempty(U)
  U=zeros(size(B,2),size(Y,2));
end

%Size checks:

%Init arrays:
Xp=nan(size(A,1),size(Y,2)+1);
X=nan(size(A,1),size(Y,2));
priorPinv=nan(size(A,1),size(A,1),size(Y,2)+1);
Pinv=nan(size(A,1),size(A,1),size(Y,2));
rejSamples=zeros(size(Y));

%Priors:
prevX=x0;
prevPinv=P0inv;
Xp(:,1)=x0;
priorPinv(:,:,1)=P0inv;

CtRinv=lsqminnorm(R,C,tol)';
CtRinvC=CtRinv*C;
Y_D=Y-D*U;
Qinv=pinv(Q,tol);

%Do the filtering
if ~outlierRejection
    for i=1:size(Y,2)
    %First, do the update given the output at this step:
      [prevX,prevPinv]=KFupdateEff(CtRinv,CtRinvC,prevX,prevPinv,Y_D(:,i)); %More efficient implementation
      %[prevX,prevP]=KFupdate(C,R,prevX,prevP,Y(:,i),d);
      X(:,i)=prevX;
      Pinv(:,:,i)=prevPinv;
      %Then, predict next step:
      b=B*U(:,i);
      %[prevX,prevP]=KFpredict(A,Q,prevX,prevP,b);
      [prevX,prevPinv]=KFpredictEff(A,Qinv,prevX,prevPinv,b);
      Xp(:,i+1)=prevX;
      priorPinv(:,:,i+1)=prevPinv;
    end
else
    error('Outlier rejection NOT implemented in efficient mode')
end


end
