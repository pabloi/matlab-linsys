%%
addpath(genpath('../')) %Adding the matlab-sysID toolbox to path, just in case
%% Create model:
D1=2;
D2=100;
N=1000;
A=randn(D1);
A=.98*A./max(abs(eig(A))); %Setting the max eigenvalue to .98
A=[.95,0;0,.99];
A=jordan(A); %Using A in its jordan canonical form so we can compare identified systems, WLOG
%B=3*randn(D1,1);
%B=B./sign(B); %Forcing all elements of B to be >0, WLOG
B=(eye(size(A))-A)*ones(size(A,1),1); %WLOG, arbitrary scaling
U=[zeros(300,1);ones(N,1);zeros(N/2,1)]'; %Step input and then removed
C=randn(D2,D1);
D=randn(D2,1);
Q=eye(D1)*.0005;
R=eye(D2)*.01;

%% Simulate
NN=size(U,2);
x0=zeros(D1,1);
[Y,X]=fwdSim(U,A,B,C,D,x0,Q,R);
[Y1,X1]=fwdSim(U,A,B,C,D,x0,[],[]); %Noiseless simulation, for comparison
[Xs,Ps]=statKalmanSmoother(Y,A,C,Q,R,[],[],B,D,U,false); %Kalman smoother estimation of states, given the true parameters (this is the best possible estimation of states)
logL=dataLogLikelihood(Y,U,A,B,C,D,Q,R,Xs,Ps)
%%
Nreps=10;
D1=2;
method='fast';
[Ae,Be,Ce,De,Qe,Re,Xe,Pe]=randomStartEM(Y,U,D1,Nreps,method);
logl=dataLogLikelihood(Y,U,Ae,Be,Ce,De,Qe,Re,Xe)

%TODO: keep some record of all local optima, and compare them: are they
%similar solutions?
%%
