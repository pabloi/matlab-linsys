%%
addpath(genpath('../EM/')) %Adding the matlab-sysID toolbox to path, just in case
addpath(genpath('../kalman/'))
addpath(genpath('../aux/'))
addpath(genpath('../sim/'))
addpath(genpath('../../robustCov/'))
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
[Xs,Ps]=statKalmanSmoother(Y,A,C,Q,R,[],[],B,D,U,false); %Kalman smoother estimation of states, given the true parameters (this is the best possible estimation of states)
logL=dataLogLikelihood(Y,U,A,B,C,D,Q,R,Xs(:,1),Ps(:,:,1))

%% fast EM estimation
tic
[Ah1,Bh1,Ch1,Dh1,Qh1,Rh1,Xh1,Ph1]=EM(Y,U,2,[],1);
logLh1=dataLogLikelihood(Y,U,Ah1,Bh1,Ch1,Dh1,Qh1,Rh1,Xh1(:,1),Ph1(:,:,1))
toc
%[Ah1,Bh1,Ch1,Xh1,~,Qh1] = canonizev2(Ah1,Bh1,Ch1,Xh1,Qh1);
%[Ah1,Bh1,Ch1,Xh1,~,Ph1(:,:,1)] = canonizev2(Ah1,Bh1,Ch1,Xh1,Ph1(:,:,1));
[Xs1,Ps1,Pt1,~,~,Xp1,Pp1]=statKalmanSmoother(Y,Ah1,Ch1,Qh1,Rh1,Xh1(:,1),Ph1(:,:,1),Bh1,Dh1,U,false);
[Xs3,Ps3,Pt3,~,~,Xp3,Pp3]=statKalmanSmootherFast(Y,Ah1,Ch1,Qh1,Rh1,Xh1(:,1),Ph1(:,:,1),Bh1,Dh1,U,false);
%% Do QR refinement
[Qopt,Ropt]=refineQR(Y-Ch1*Xp1(:,1:end-1)-Dh1*U,Pp1,Ch1,Ah1,Rh1);
%% See that the estimated states did not change
[Xs2,Ps2,Pt2,~,~,Xp2,Pp2]=statKalmanSmoother(Y,Ah1,Ch1,Qopt,Ropt,Xh1(:,1),Ph1(:,:,1),Bh1,Dh1,U,false);
figure;
plot(Xp1');
hold on
set(gca,'ColorOrderIndex',1)
 plot(Xp2','--');
%% See the logL improved nonetheless
logLh2=dataLogLikelihood(Y,U,Ah1,Bh1,Ch1,Dh1,Qopt,Ropt,Xh1(:,1),Ph1(:,:,1))
