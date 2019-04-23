%%
addpath(genpath('../../')) %Adding the matlab-sysID toolbox to path, just in case
%%
clear all
%% Create model:
D1=3;
D2=100;
N=1000;
A=diag(1-.1*abs(rand(D1,1)));
%A=[.97,0;0,.995];
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

%% Identify 1alt: trueEM starting from non-true solution
opts.fastFlag=0;
opts.robustFlag=0;
tic
[J,B,C,D,Q,R,X,P,logL,outlog]=EM(Y,U,D1,opts);
toc
model{1}=autodeal(J,B,C,D,X,Q,R,logL);
model{1}.name='EM (single run, normal)';
%%
opts.robustFlag=false;
tic
[J,B,C,D,Q,R,X,P,logL,outlog]=EM2(Y,U,D1,opts);
toc
model{2}=autodeal(J,B,C,D,X,Q,R,logL);
model{2}.name='EM (fast)';
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
