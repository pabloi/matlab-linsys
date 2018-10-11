%%
addpath(genpath('../EM/')) %Adding the matlab-sysID toolbox to path, just in case
addpath(genpath('../kalman/'))
addpath(genpath('../aux/'))
addpath(genpath('../sim/'))
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
%% Median-filtered
 binw=3;
 Y2=[medfilt1(Y(:,1:300),binw,[],2,'truncate'), medfilt1(Y(:,301:300+N),binw,[],2,'truncate'), medfilt1(Y(:,[301+N:end]),binw,[],2,'truncate')];
%% Moving average filtered:
 binw=3;
 Y3=[conv2(Y(:,1:300),ones(1,binw)/binw,'same'), conv2(Y(:,301:300+N),ones(1,binw)/binw,'same'), conv2(Y(:,[301+N:end]),ones(1,binw)/binw,'same')];
 Y3(:,1:2)=Y(:,1:2);
 Y3(:,299:302)=Y(:,299:302);
 Y3(:,299+N:302+N)=Y(:,299+N:302+N);
 
 %% Store true model"\:
 J=A;
 model{1}=autodeal(J,B,C,D,X,Q,R,logL);
 model{1}.name='True';
%% Identify 1alt: trueEM starting from non-true solution
opts.fastFlag=0;
opts.robustFlag=true;
tic
[A,B,C,D,Q,R,X,P]=EM(Y,U,D1,opts);
logL=dataLogLikelihood(Y,U,A,B,C,D,Q,R,X(:,1),P(:,:,1));
toc
[J,B,C,X,~,Q] = canonizev2(A,B,C,X,Q);
model{2}=autodeal(J,B,C,D,X,Q,R,logL);
model{2}.name='EM (fast,robust)';
%%
opts.fastFlag=0;
opts.robustFlag=true;
tic
[A,B,C,D,Q,R,X,P]=EM(Y2,U,D1,opts);
logL=dataLogLikelihood(Y,U,A,B,C,D,Q,R,X(:,1),P(:,:,1));
toc
[J,B,C,X,fV,Q] = canonizev2(A,B,C,X,Q);
model{3}=autodeal(J,B,C,D,X,Q,R,logL);
model{3}.name='EM (fast,robust, median filtered)';
%%
opts.fastFlag=0;
opts.robustFlag=true;
tic
[A,B,C,D,Q,R,X,P]=EM(Y3,U,D1,opts);
logL=dataLogLikelihood(Y,U,A,B,C,D,Q,R,X(:,1),P(:,:,1));
toc
[J,B,C,X,fV,Q] = canonizev2(A,B,C,X,Q);
model{4}=autodeal(J,B,C,D,X,Q,R,logL);
model{4}.name='EM (fast,robust, linear filtered)';
%%
%figure; hold on; plot(Y'); plot(Y2'); plot(Y3')
%% COmpare
compareModels(model,Y,U)
%% COmpare
%compareModels(model,Y2,U)