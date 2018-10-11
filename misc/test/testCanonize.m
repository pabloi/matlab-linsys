%%
addpath(genpath('../EM/')) %Adding the matlab-sysID toolbox to path, just in case
addpath(genpath('../kalman/'))
addpath(genpath('../aux/'))
addpath(genpath('../sim/'))
addpath(genpath('../../robustCov/'))
%% Create model:
clearvars
D1=2;
D2=180;
N=1000;
A=randn(D1);
A=[.97,0;0,.995];
A=jordan(A); %Using A in its jordan canonical form so we can compare identified systems, WLOG
%B=3*randn(D1,1);
%B=B./sign(B); %Forcing all elements of B to be >0, WLOG
B=(eye(size(A))-A)*ones(size(A,1),1); %WLOG, arbitrary scaling
U=[zeros(300,1);ones(N,1);zeros(N/2,1)]'; %Step input and then removed
C=randn(D2,D1);
D=randn(D2,1);
Q=eye(D1)*.0001;
R=eye(D2)*.01;

%% Simulate many times:
NN=size(U,2);
x0=randn(D1,1);
[Y,X]=fwdSim(U,A,B,C,D,x0,Q,R);
P0=eye(size(Q));
logL=dataLogLikelihood(Y,U,A,B,C,D,Q,R,x0,P0)
%% Canonize v2:
[A,B,C,x0,~,Q,P0] = canonizev3(A,B,C,x0,Q,P0);
logL=dataLogLikelihood(Y,U,A,B,C,D,Q,R,x0,P0)
%% Canonize v3:
[A,B,C,x0,~,Q,P0] = canonizev3(A,B,C,x0,Q,P0);
logL=dataLogLikelihood(Y,U,A,B,C,D,Q,R,x0,P0)