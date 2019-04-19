%% Test sqrtUpdate vs. conventional approach

%% Generate system
Ny=4;
Nx=4;
cR=randn(Ny);
R=cR*cR';
cQ=randn(Nx);
Q=cQ*cQ';
cholQ=chol(Q);
A=diag(rand(Nx,1).^.1);
C=randn(Ny,Nx);

%% Generate state and uncertainty for 1 step
y=randn(Ny,1);
prevX=randn(Nx,1);
cP=randn(Nx);
prevP=cP*cP';
cholR=chol(R);
cholP=chol(prevP);
b=rand(Nx,1);

%% Traditional update
tic
for i=1:1000
[newX,newP,logL]=KFupdate(C,R,y,prevX,prevP,0);
[newX1,newP1]=KFpredict(A,Q,newX,newP,b);
end
toc
newCholP1=chol(newP1);
%% sqrtUpdate:
tic
for i=1:1000
[newX2,newCholP2] = sqrtUpdate(y',prevX', cholP, cholR, cholQ, A', C',b');
end
toc

%% Difference:
newX1-newX2'
newCholP1'*newCholP1 - newCholP2'*newCholP2