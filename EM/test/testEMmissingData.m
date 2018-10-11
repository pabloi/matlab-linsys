%%
addpath(genpath('../EM/')) %Adding the matlab-sysID toolbox to path, just in case
addpath(genpath('../kalman/'))
addpath(genpath('../aux/'))
addpath(genpath('../sim/'))
addpath(genpath('../../robustCov'))
%% Create model:
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
[A,B,C,~,~,Q] = canonizev2(A,B,C,Q,Q);
%% Simulate
NN=size(U,2);
x0=zeros(D1,1);
[Y,X]=fwdSim(U,A,B,C,D,x0,Q,R);
[Y1,X1]=fwdSim(U,A,B,C,D,x0,[],[]); %Noiseless simulation, for comparison

%% Best states, given true params
[Xs,Ps]=statKalmanSmoother(Y,A,C,Q,R,[],[],B,D,U,false); %Kalman smoother estimation of states, given the true parameters (this is the best possible estimation of states)
logL=dataLogLikelihood(Y,U,A,B,C,D,Q,R,Xs(:,1),Ps(:,:,1))
%% Best params, given true states
[A1,B1,C1,D1,Q1,R1,x01,P01]=estimateParams(Y,U,X(:,1:end-1),repmat(1e-8*eye(2),1,1,size(U,2)),repmat(1e-9*eye(2),1,1,size(U,2))) ;
logL1=dataLogLikelihood(Y,U,A1,B1,C1,D1,Q1,R1,X(:,1),1e-8*eye(2))

%% Identify 1alt: trueEM starting from true solution
tic
[Ah,Bh,Ch,Dh,Qh,Rh,Xh,Ph]=randomStartEM(Y,U,2,10);
logLh=dataLogLikelihood(Y,U,Ah,Bh,Ch,Dh,Qh,Rh,Xh(:,1),Ph(:,:,1))
toc
[Ah,Bh,Ch,Xh,V,Qh] = canonizev2(Ah,Bh,Ch,Xh,Qh);
%% Remove 5% data:
aux=rand(1,size(Y,2))>.99;
Y2=Y;
Y2(:,aux)=NaN;
%% Identify 1alt: trueEM starting from true solution
tic
[fAh,fBh,fCh,fDh,fQh,fRh,fXh,fPh]=randomStartEM(Y2,U,2,10);
flogLh=dataLogLikelihood(Y,U,fAh,fBh,fCh,fDh,fQh,fRh,fXh(:,1),fPh(:,:,1))
toc
[fAh,fBh,fCh,fXh,fV,fQh] = canonizev2(fAh,fBh,fCh,fXh,fQh);
