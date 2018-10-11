%%
addpath(genpath('../EM/')) %Adding the matlab-sysID toolbox to path, just in case
addpath(genpath('../kalman/'))
addpath(genpath('../aux/'))
addpath(genpath('../sim/'))
addpath(genpath('../../robustCov/'))
%%
clear all
%% Create model:
D1=2;
D2=2;
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
%% Identify 1alt: trueEM starting from true solution
% fastFlag=0;
% tic
% [fAh,fBh,fCh,fDh,fQh,fRh,fXh,fPh]=EM(Y,U,Xs,[],fastFlag);
% flogLh=dataLogLikelihood(Y,U,fAh,fBh,fCh,fDh,fQh,fRh,fXh(:,1),fPh(:,:,1))
% toc
% [fAh,fBh,fCh,fXh,fV,fQh] = canonizev2(fAh,fBh,fCh,fXh,fQh);
%% Identify 1alt: trueEM starting from non-true solution
opts.fastFlag=0;
opts.robustFlag=true;
tic
[A,B,C,D,Q,R,X,P]=EM(Y,U,D1,opts);
logL=dataLogLikelihood(Y,U,A,B,C,D,Q,R,X(:,1),P(:,:,1));
toc
[J,B,C,X,~,Q,P] = canonizev2(A,B,C,X,Q,P);
model{1}=autodeal(J,B,C,D,X,Q,R,logL);
model{1}.name='EM (fast,robust)';
%%
opts.robustFlag=false;
tic
[A,B,C,D,Q,R,X,P]=randomStartEM(Y,U,2,5,opts);
logL=dataLogLikelihood(Y,U,A,B,C,D,Q,R,X(:,1),P(:,:,1));
toc
[J,B,C,X,~,Q,P] = canonizev2(A,B,C,X,Q,P);
model{2}=autodeal(J,B,C,D,X,Q,R,logL);
model{2}.name='EM (itered,fast)';
%%
% LDS.A=Ah;
% LDS.B=Bh;
% LDS.C=Ch;
% LDS.D=Dh;
% LDS.Q=Qh;
% LDS.R=Rh;
% LDS.x0=Xh(:,1);
% LDS.V0=Ph(:,:,1);
% llh=LikelihoodLDS(LDS,Y,U);
%% Cheng & Sabes %Too slow
% addpath(genpath('../ext/lds-1.0/'))
% LDS.A=eye(size(A));
% LDS.B=ones(size(B));
% LDS.C=randn(D2,D1);
% LDS.D=randn(D2,1);
% LDS.Q=eye(size(Q));
% LDS.R=eye(size(R));
% LDS.x0=zeros(D1,1);
% LDS.V0=1e8 * eye(size(A)); %Same as my smoother uses
% [LDS,Lik,Xcs,Vcs] = IdentifyLDS(2,Y,U,U,LDS);
% csLogLh=dataLogLikelihood(Y,U,LDS.A,LDS.B,LDS.C,LDS.D,LDS.Q,LDS.R,LDS.x0,LDS.V0)
% toc

%% Compare
compareModels(model,Y,U)