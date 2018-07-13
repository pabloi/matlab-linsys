%%
addpath(genpath('../aux/'))
%% Create model:
D1=2;
D2=100;
N=1000;
A=randn(D1);
A=.98*A./max(abs(eig(A))); %Setting the max eigenvalue to .98
A=[.99,0;0,.95];
A=jordan(A); %Using A in its jordan canonical form so we can compare identified systems, WLOG
B=3*randn(D1,1);
B=B./sign(B); %Forcing all elements of B to be >0, WLOG
U=[zeros(300,1);ones(N,1);zeros(N/2,1)]'; %Step input and then removed
C=randn(D2,D1);
C=C./sqrt(sum(C.^2,1));
D=randn(D2,1);
X0=randn(2,1);
Q=eye(D1)*.9;
R=eye(D2)*.4;

%% Simulate
NN=size(U,2);
x0=zeros(D1,1);
[Y,X]=fwdSim(U,A,B,C,D,x0,Q,R);

%% Identify 2: we do not know C,D (we do know D1);
[Ah,Bh,Ch,Dh,Qh,Rh,Xh]=fastEM(Y,U,2);
[J,K,Ch,Xh,V] = canonizev2(Ah,Bh,Ch,Xh);
ss=sign(K);
K=K.*ss;
Xh=Xh.*ss;
J=ss'*K*ss;


figure;
subplot(2,1,1)
hold on
plot(X'./sqrt(sum(X.^2,2))')
plot(Xh'./sqrt(sum(Xh.^2,2))','o')
title('States')
subplot(2,1,2)
hold on
%Zh=
%Wh=
title('Noise realizations')
