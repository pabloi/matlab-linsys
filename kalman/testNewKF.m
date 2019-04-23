%addpath(genpath('../'))
%% Create model:
D1=5;
D2=100;
A=diag(rand(D1,1));
A=.9999*A; %Setting the max eigenvalue to .9999
B=(eye(size(A))-A)*ones(size(A,1),1); %WLOG, arbitrary scaling so all states asymptote at 1
N=2000;
U=[zeros(300,1);ones(N,1);zeros(N/2,1)]'; %Step input and then removed
C=randn(D2,D1);
D=randn(D2,1);
Q=eye(D1)*1e-3;
R=1e-2*eye(D2); %CS2006 performance degrades (larger state estimation errors) for very small R

%% Simulate
NN=size(U,2);
x0=zeros(D1,1);
[Y,X]=fwdSim(U,A,B,C,D,x0,Q,R);
[Y1,X1]=fwdSim(U,A,B,C,D,x0,[],[]); %Noiseless simulation, for comparison
X=X(:,1:end-1);

%% Define initial uncertainty
P0=zeros(D1);
P0=diag(Inf*ones(D1,1));
P0=1e1*eye(D1);

%% Do kalman filter standard
tic
mdl=1;
opts.noReduceFlag=false;
opts.fastFlag=0;
[Xf,Pf,Xp,Pp,~,logL(mdl)]=statKalmanFilter(Y,A,C,Q,R,x0,P0,B,D,U,opts); %Kalman smoother estimation of states, given the true parameters (this is the best possible estimation of states)
Xf1=Xf;
Pf1=Pf;
res(mdl)=norm(Xf-X,'fro')^2;
maxRes(mdl)=max(sum((Xf-X).^2));
name{mdl}='KF1';
tc(mdl)=toc;
%% Do kalman filter fast
tic
mdl=2;
opts.noReduceFlag=false;
opts.fastFlag=0;
[Xf,Pf,Xp,Pp,~,logL(mdl),S]=statKalmanFilter2(Y,A,C,Q,R,x0,P0,B,D,U,opts); %Kalman smoother estimation of states, given the true parameters (this is the best possible estimation of states)
Xf2=S*Xf;
Pf2=Pf; %Actually, a sqrtm
res(mdl)=norm(Xf2-X,'fro')^2;
maxRes(mdl)=max(sum((Xf2-X).^2));
name{mdl}='KF2';
tc(mdl)=toc;

%% Do kalman smoother standard
tic
mdl=3;
opts.noReduceFlag=false;
opts.fastFlag=0;
[Xs,Ps,Pt,Xf,Pf,Xp,Pp,~,logL(mdl)]=statKalmanSmoother(Y,A,C,Q,R,x0,P0,B,D,U,opts); %Kalman smoother estimation of states, given the true parameters (this is the best possible estimation of states)
Xs1=Xs;
Ps1=Ps;
res(mdl)=norm(Xs-X,'fro')^2;
maxRes(mdl)=max(sum((Xs-X).^2));
name{mdl}='KS1';
tc(mdl)=toc;

%% Do fast smoother
tic
mdl=4;
opts.noReduceFlag=false;
opts.fastFlag=0;
[Xs,Ps,Pt,Xf,Pf,Xp,Pp,~,logL(mdl),S]=statKalmanSmoother2(Y,A,C,Q,R,x0,P0,B,D,U,opts); %Kalman smoother estimation of states, given the true parameters (this is the best possible estimation of states)
Xs2=S*Xs;
Ps2=Ps;
res(mdl)=norm(Xs2-X,'fro')^2;
maxRes(mdl)=max(sum((Xs2-X).^2));
name{mdl}='KS2';
tc(mdl)=toc;
%%
tc
res
maxRes
%%
%figure; plot(Xf1'); hold on; plot(Xf2')
%figure; plot(Xs1'); hold on; plot(Xs2')