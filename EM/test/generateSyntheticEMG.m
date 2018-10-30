function [Y,U,A,B,C,D,Q,R,x0,Yoff]=generateSyntheticEMG(method,U,A,B,C,D,Q,R,x0,Yoff)
  if nargin<1 || isempty(method)
    method='gaussian';
  end
if nargin<3
  nx=3;
  A=diag(exp(-1./[20;100;600]));
else
  nx=size(A,1);
end
if nargin<4
  B=1-diag(A); %All states tend to 1 under input response, WLOG.
end
if nargin<5
  ny=180;
  C=.5*rand(ny,nx)-.25; %All values in -.25,.25
  for i=1:nx
    C(:,i)=reshape(conv2(reshape(C(:,i),12,15),ones(3,3)/9,'same'),ny,1); %To get some structure in columns of C
  end
else
  ny=size(C,1);
end
if nargin<6
  D=.5*rand(ny,1)-.25; %All values in -.25, .25
  D(:)=conv2(reshape(D,12,15),ones(3,3)/9,'same');
end
if nargin<1
  U=[zeros(1,150) ones(1,900) zeros(1,600)];
end
if nargin<7
  Q=.01*randn(nx);
  Q=Q*Q';
end
if nargin<8
  R=.01*randn(ny);
  R=R*R';
end
if nargin<9
  x0=zeros(nx,1);
end
if nargin<10
  Yoff=randn(ny,1);
end
%Basic method: gaussian noise
[~,X]=fwdSim([U;ones(size(U))],A,B,C,D,Yoff,x0,Q,[]); %Legs modeled with opposite trajectories and independent observation noises.
cR=mycholcov([[R, zeros(size(R))]; [zeros(size(R)), R]]);
switch method
  case 'gaussian'
    Y=C*X+D*U+Yoff + cR'*randn(size(C,1),size(X,2)); %Gaussian noise
  case 'rectified'
    Y=C*X+D*U+Yoff + cR'*randn(size(C,1),size(X,2));
    Y=abs(Y);
  case 'square'
    mL=C*X+D*U+Yoff;
    YL=mL+ (cR'*randn(size(C,1),size(X,2))).^2 - diag(R); %Mean should be mL
    mR=-C*X-D*U+Yoff;
    YR=mR + (cR'*randn(size(C,1),size(X,2))).^2 -diag(R); %Mean should be mR
  case 'gamma'
    YL=sign(Yoff+C*X+D*U).*gammarnd(abs(C*X+D*U+Yoff),1);
    YR=sign(Yoff-C*X-D*U).*gammarnd(abs(Yoff-C*X-D*U),1);
  otherwise
    error('Unrecognized method')
  end
end
end
