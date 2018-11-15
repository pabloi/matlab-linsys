%%
[folder]=fileparts(mfilename('fullpath'));
addpath(genpath([folder '/../../']))%Adding the matlab-sysID toolbox to path, just in case
addpath(genpath([folder '/../../../robustCov/'])) %Adding robust Cov toolbox
%%
clearvars
warning('off')
%% Create model:
D1=2;
D2=180;
N=1000;
A=[.97,0;0,.995];
B=(eye(size(A))-A)*ones(size(A,1),1); %WLOG, arbitrary scaling
U=[zeros(300,1);ones(N,1);zeros(N/2,1)]'; %Step input and then removed
C=randn(D2,D1);
D=randn(D2,1);
Q=eye(D1)*.0001;
R=eye(D2)*.01;
[A,B,C,~,~,Q] = canonize(A,B,C,Q,Q);

%% Simulate
NN=size(U,2);
x0=zeros(D1,1);
[Y,X]=fwdSim(U,A,B,C,D,x0,Q,R);
[Y1,X1]=fwdSim(U,A,B,C,D,x0,[],[]); %Noiseless simulation, for comparison
[Xs,Ps,~,~,~,~,~,~,logL]=statKalmanSmoother(Y,A,C,Q,R,[],[],B,D,U,[]); %Kalman smoother estimation of states, given the true parameters (this is the best possible estimation of states)
trueTau=-1./log(eig(A))
logL
%% EM - classic
tic
opts.Niter=500;
opts.fastFlag=false;
[Ah1,Bh1,Ch1,Dh1,Qh1,Rh1,Xh1,Ph1,logL1]=EM(Y,U,2,opts);
toc
[Ah1,Bh1,Ch1,Xh1,~,Qh1] = canonize(Ah1,Bh1,Ch1,Xh1,Qh1);
logL1

%% EM - fast
tic
opts.fastFlag=true; %Self-selecting number of samples for transient (normal) filtering
[Ah2,Bh2,Ch2,Dh2,Qh2,Rh2,Xh2,Ph2,logL2]=EM(Y,U,2,opts);
toc
[Ah2,Bh2,Ch2,Xh2,~,Qh2] = canonize(Ah2,Bh2,Ch2,Xh2,Qh2);
logL2

%% EM - fast, forced
tic
opts.fastFlag=30; %Doing normal filtering for 30 strides, and then assuming steady-state was reached
[Ah3,Bh3,Ch3,Dh3,Qh3,Rh3,Xh3,Ph3,logL3]=EM(Y,U,2,opts);
toc
[Ah3,Bh3,Ch3,Xh3,~,Qh3] = canonize(Ah3,Bh3,Ch3,Xh3,Qh3);
logL3

