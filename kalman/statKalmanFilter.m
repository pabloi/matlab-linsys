function [X,P,Xp,Pp,rejSamples]=statKalmanFilter(Y,A,C,Q,R,x0,P0,B,D,U,outlierRejection)
%filterStationary implements a Kalman filter assuming
%stationary (fixed) noise matrices and system dynamics
%The model is: x[k+1]=A*x[k]+b+v[k], v~N(0,Q)
%y[k]=C*x[k]+d+w[k], w~N(0,R)
%And X[0] ~ N(x0,P0) -> Notice that this is different from other
%implementations, where P0 is taken to be cov(x[0|-1]) so x[0]~N(x0,A*P0*A'+Q)
%See for example Ghahramani and Hinton 1996

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

%Size checks:
%TODO

%Init arrays:
if isa(Y,'gpuArray') %For code to work on gpu
    Xp=nan(size(A,1),size(Y,2)+1,'gpuArray');
    X=nan(size(A,1),size(Y,2),'gpuArray');
    Pp=nan(size(A,1),size(A,1),size(Y,2)+1,'gpuArray');
    P=nan(size(A,1),size(A,1),size(Y,2),'gpuArray');
    rejSamples=zeros(size(Y),'gpuArray');
else
    Xp=nan(size(A,1),size(Y,2)+1);
    X=nan(size(A,1),size(Y,2));
    Pp=nan(size(A,1),size(A,1),size(Y,2)+1);
    P=nan(size(A,1),size(A,1),size(Y,2));
    rejSamples=zeros(size(Y));
end

%Priors:
tol=1e-8;
prevX=x0;
prevP=P0;
%prevsP=decomposition(prevP,'chol','upper');
%previP=pinv(P0,tol);
Xp(:,1)=x0;
Pp(:,:,1)=P0;

%sR=chol(R); %cholesky decomposition
%isR=(sR\eye(size(sR)))'; %Cholesky decomp of Rinv
%aux=C'*isR';
%CtRinv=aux*isR;
%CtRinvC=aux*aux'; %Ensures psd

%CtRinv=lsqminnorm(R,C,tol)';  %Equivalent to C'/R;, not gpu ready
CtRinv=C'*pinv(R,tol); %gpu-ready
CtRinvC=CtRinv*C;

Y_D=Y-D*U;
CtRinvY=CtRinv*Y_D;
BU=B*U;
%iQ=pinv(Q,tol);

%Do the filtering
for i=1:size(Y,2)
  %First, do the update given the output at this step:
  if ~outlierRejection
    [prevX,prevP]=KFupdate(CtRinvY(:,i),CtRinvC,prevX,prevP);
    %[prevX,prevsP,prevP]=KFupdatev2(CtRinvY(:,i),CtRinvC,prevX,prevsP);
    %[prevX,previP]=KFupdateEff(CtRinvY(:,i),CtRinvC,prevX,previP);
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
  %[prevX,prevsP,prevP]=KFpredictv2(A,Q,prevX,prevP,BU(:,i));
  %[prevX,previP]=KFpredictEff(A,iQ,prevX,previP,BU(:,i));
  Xp(:,i+1)=prevX;
  Pp(:,:,i+1)=prevP;
end

end
