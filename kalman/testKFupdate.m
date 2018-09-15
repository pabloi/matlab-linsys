%Test KFupdate
addpath(genpath('../aux/'))
%% size(R)>size(P), invertible P
C=randn(180,3);
Rs=randn(180);
R=Rs'*Rs;
x0=rand(3,1);
P0s=randn(3);
P0=P0s*P0s';
y=randn(180,1);

cR=chol(R);
icR=cR\eye(size(R));
CtcR=C'*icR;
CtRinvC=CtcR*CtcR';
CtRinvY=(C'*(icR*icR'))*y;

Nreps=1e3;
tic; for i=1:Nreps; [x1,P1]=KFupdate(C,R,y,x0,P0); end; toc;
tic; for i=1:Nreps; [x,P]=KFupdateAlt(CtRinvY,CtRinvC,x0,P0); end; toc %This should be faster

norm(x-x1)
norm(P-P1,'fro')

%% size(R)>size(P), non-invertible P
C=randn(180,3);
Rs=randn(180);
R=Rs'*Rs;
x0=rand(3,1);
P0s=randn(3,2);
P0=P0s*P0s';
y=randn(180,1);

cR=chol(R);
icR=cR\eye(size(R));
CtcR=C'*icR;
CtRinvC=CtcR*CtcR';
CtRinvY=(C'*(icR*icR'))*y;

Nreps=1e3;
tic; for i=1:Nreps; [x1,P1]=KFupdate(C,R,y,x0,P0); end; toc;
tic; for i=1:Nreps; [x,P]=KFupdateAlt(CtRinvY,CtRinvC,x0,P0); end; toc

norm(x-x1)
norm(P-P1,'fro')

%% size(P)>size(R)
C=randn(3,100);
Rs=randn(3);
R=Rs'*Rs;
x0=rand(100,1);
P0s=randn(100);
P0=P0s*P0s';
y=randn(3,1);

cR=chol(R);
icR=cR\eye(size(R));
CtcR=C'*icR;
CtRinvC=CtcR*CtcR';
CtRinvY=(C'*(icR*icR'))*y;

Nreps=1e3;
tic; for i=1:Nreps; [x1,P1]=KFupdate(C,R,y,x0,P0); end; toc; %This should be faster
tic; for i=1:Nreps; [x,P]=KFupdateAlt(CtRinvY,CtRinvC,x0,P0); end; toc

norm(x-x1)
norm(P-P1,'fro')

%% Update depends only on C*pinv(C)*y, and not on y itself (only matters for size(R)>size(P) or rank(C)<dim(C) otherwise C*pinv(C)=I)

C=randn(180,3);
Rs=randn(180);
R=Rs'*Rs;
x0=rand(3,1);
P0s=randn(3);
P0=P0s*P0s';
y=randn(180,1);

cR=chol(R);
icR=cR\eye(size(R));
CtcR=C'*icR;
CtRinvC=CtcR*CtcR';
CtRinvY=(C'*(icR*icR'))*y;

z=(C*pinv(C))*y;
CtRinvZ=(C'*(icR*icR'))*z;

Nreps=1e2;
tic; for i=1:Nreps; [x1,P1]=KFupdate(C,R,y,x0,P0); end; toc; %This should be faster
tic; for i=1:Nreps; [x2,P2]=KFupdate(C,R,z,x0,P0); end; toc; %This should be faster
tic; for i=1:Nreps; [x,P]=KFupdateAlt(CtRinvY,CtRinvC,x0,P0); end; toc
tic; for i=1:Nreps; [x,P]=KFupdateAlt(CtRinvZ,CtRinvC,x0,P0); end; toc

norm(x2-x1)
norm(P2-P1,'fro')
norm(x-x1)
norm(P-P1,'fro')