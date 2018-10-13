%test dataLogLikelihood

%%
addpath(genpath('./')) %Adding the matlab-sysID toolbox to path, just in case
addpath(genpath('../robustCov/'))
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
[Xs,Ps,~,~,~,Xp,Pp]=statKalmanSmoother(Y,A,C,Q,R,[],[],B,D,U,false); %Kalman smoother estimation of states, given the true parameters (this is the best possible estimation of states)

%% Compare likelihood estimates:
%% Exact
tic
logLexact=dataLogLikelihood(Y,U,A,B,C,D,Q,R,Xp(:,1),Pp(:,:,1),'exact')
toc

%% Approx
tic
logLapp=dataLogLikelihood(Y,U,A,B,C,D,Q,R,Xp(:,1),Pp(:,:,1),'approx')
toc

%% Fast
tic
logLfast=dataLogLikelihood(Y,U,A,B,C,D,Q,R,Xp(:,1),Pp(:,:,1),'fast')
toc

%% Max
tic
logLmax=dataLogLikelihood(Y,U,A,B,C,D,Q,R,Xp(:,1),Pp(:,:,1),'max')
toc
%%
tic
%[A1,B1,C1,D1,Q1,R1,X1,P1]=EM(Y,U,2,[],1);
toc
%[Xs1,Ps1,~,~,~,Xp1,Pp1]=statKalmanSmoother(Y,A1,C1,Q1,R1,[],[],B1,D1,U,false); %
tic
%logLapp1=dataLogLikelihood(Y,U,A1,B1,C1,D1,Q1,R1,Xp1(:,1),Pp1(:,:,1),'approx')
%logLmax1=dataLogLikelihood(Y,U,A1,B1,C1,D1,Q1,R1,Xp1(:,1),Pp1(:,:,1),'max')
toc
%% test QR refinement
% [Qo,Ro]=refineQR(Y-C*Xp(:,1:end-1)-D*U,Pp,C,A,R);
% tic
% logLapp2=dataLogLikelihood(Y,U,A1,B1,C1,D1,Qo,Ro,Xp1(:,1),Pp1(:,:,1),'approx')
% logLapp2alt=dataLogLikelihood(Y,U,A1,B1,C1,D1,Qo,Ro,Xp1,Pp1,'approx')
% toc
