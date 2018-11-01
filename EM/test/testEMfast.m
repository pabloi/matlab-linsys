%%
addpath(genpath('../EM/')) %Adding the matlab-sysID toolbox to path, just in case
addpath(genpath('../kalman/'))
addpath(genpath('../misc/'))
addpath(genpath('../../robustCov/'))
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
[Xs,Ps]=statKalmanSmoother(Y,A,C,Q,R,[],[],B,D,U,[]); %Kalman smoother estimation of states, given the true parameters (this is the best possible estimation of states)
logL=dataLogLikelihood(Y,U,A,B,C,D,Q,R,Xs(:,1),Ps(:,:,1))

%% randomStartEM - classic
tic
[Ah1,Bh1,Ch1,Dh1,Qh1,Rh1,Xh1,Ph1]=randomStartEM(Y,U,2,1,[]);
logLh1=dataLogLikelihood(Y,U,Ah1,Bh1,Ch1,Dh1,Qh1,Rh1,Xh1(:,1),Ph1(:,:,1));
toc
[Ah1,Bh1,Ch1,Xh1,~,Qh1] = canonize(Ah1,Bh1,Ch1,Xh1,Qh1);

%% randomStartEM - fast
tic
opts.fastFlag=true;
[Ah2,Bh2,Ch2,Dh2,Qh2,Rh2,Xh2,Ph2]=randomStartEM(Y,U,2,1,opts);
logLh2=dataLogLikelihood(Y,U,Ah2,Bh2,Ch2,Dh2,Qh2,Rh2,Xh2(:,1),Ph2(:,:,1));
toc
[Ah2,Bh2,Ch2,Xh2,~,Qh2] = canonize(Ah2,Bh2,Ch2,Xh2,Qh2);

%% Plot results
