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

%% Dim red: Update depends only on z=C'*inv(R)*y, and not on y itself 
%(only matters for rank(C'*inv(R))<dim(y), as it implies dim reduction)

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
Rnew=CtcR*CtcR';
z=(C'*(icR*icR'))*y;

Nreps=1e2;
tic; for i=1:Nreps; [x1,P1]=KFupdate(C,R,y,x0,P0); end; toc; 
tic; for i=1:Nreps; [x2,P2]=KFupdate(Rnew,Rnew,z,x0,P0); end; toc; 
tic; for i=1:Nreps; [x,P]=KFupdateAlt(z,Rnew,x0,P0); end; toc %This works the same IF P is invertible
%tic; for i=1:Nreps; [x,P]=KFupdateAlt(CtRinvZ,CtRinvC,x0,P0); end; toc

norm(x2-x1)
norm(P2-P1,'fro')
norm(x-x1)
norm(P-P1,'fro')

%% Dim red: non invertible R

C=randn(180,3);
Rs=randn(180,180);
R=Rs*Rs';
x0=rand(3,1);
P0s=randn(3,2);
P0=P0s*P0s';
y=randn(180,1);

cR=mycholcov(R);
icR=eye(size(R))/cR;
CtcR=C'/cR;
Rnew=CtcR*CtcR';
z=(C'*(icR*icR'))*y;

Nreps=1e2;
tic; for i=1:Nreps; [x1,P1]=KFupdate(C,R,y,x0,P0); end; toc; 
tic; for i=1:Nreps; [x2,P2]=KFupdate(Rnew,Rnew,z,x0,P0); end; toc; 
tic; for i=1:Nreps; [x,P]=KFupdateAlt(z,Rnew,x0,P0); end; toc
%tic; for i=1:Nreps; [x,P]=KFupdateAlt(CtRinvZ,CtRinvC,x0,P0); end; toc

norm(x2-x1)
norm(P2-P1,'fro')
norm(x-x1)
norm(P-P1,'fro')