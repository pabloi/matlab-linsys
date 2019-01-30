%%
addpath(genpath('../')) %Adding the matlab-sysID toolbox to path, just in case
addpath(genpath('../../robustCov/'))
%%
clear all
%% Create model:
D1=2;
D2=2;
N=1000;
A=randn(D1);
A=[.97,0;0,.995];
%B=3*randn(D1,1);
%B=B./sign(B); %Forcing all elements of B to be >0, WLOG
B=(eye(size(A))-A)*ones(size(A,1),1); %WLOG, arbitrary scaling
U=[zeros(300,1);ones(N,1);zeros(N/2,1)]'; %Step input and then removed
C=randn(D2,D1);
D=randn(D2,1);
Q=eye(D1)*.0001;
R=eye(D2)*.01;
[J,B,C,~,~,Q,~] = canonize(A,B,C,[],Q,[]);
x0=zeros(D1,1);
[Y,X]=fwdSim(U,A,B,C,D,x0,[],[]); %Noiseless simulation, for comparison
logL=dataLogLikelihood(Y,U,A,B,C,D,Q,R,[],[]);
model{1}=autodeal(J,B,C,D,X,Q,R,logL,Y);
model{1}.name='EM (fast,robust)';
%% Simulate
NN=size(U,2);
Nsubs=15;
clear X Y
for i=1:Nsubs
[Y{i},X{i}]=fwdSim(U,A,B,C,D,x0,Q,R);
Uall{i}=U;
end
%% Identify: all together
opts.fastFlag=true;
opts.robustFlag=false;
tic
[A,B,C,D,Q,R,X,P,logL]=EM(Y,Uall,D1,opts);
toc
[J,B,C,X,~,Q,P] = canonize(A,B,C,X,Q,P);
model{2}=autodeal(J,B,C,D,X,Q,R,logL);
model{2}.name='EM (fast,robust)';

%% Identify: each individually:
for i=1:Nsubs
tic
[A,B,C,D,Q,R,X,P,logL]=EM(Y{i},U,D1,opts);
toc
[J,B,C,X,~,Q,P] = canonize(A,B,C,X,Q,P);
model{2+i}=autodeal(J,B,C,D,X,Q,R,logL);
model{2+i}.name=['EM (fast,robust), sub' num2str(i)];
end
%% Compare
compareModels(model,Y,U)