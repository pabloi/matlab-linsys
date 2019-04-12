%% Script to test which of the model order selection criteria works best in practice
clear all
close all
addpath(genpath('../../'))
%% Step 1: simulate a high-dim model
load ../../../EMG-LTI-SSM/res/allDataModels.mat
model=model{5}; %4-th order model as ground truth
initC=initCond(zeros(4,1),zeros(4));
deterministicFlag=false;
noiselessModel=model;
noiselessModel.R=zeros(size(model.R));
[simDatSetNoiseless,stateE]=noiselessModel.simulate(datSet.in,initC,deterministicFlag);
cR=chol(model.R);
D2=size(model.C,1);
N=size(simDatSetNoiseless.out,2);
noise=cR*randn(D2,N);
simDatSetFixedNoise=dset(simDatSetNoiseless.in,simDatSetNoiseless.out+noise);

%% Step 2: generate a similar model but with decreasing noise levels
scaleFactor=[ones(1,150) [1500:-1:1]/1500+1];
noise=nan(D2,N);
for i=1:N
    noise(:,i)=sqrt(scaleFactor(i))*cR'*randn(D2,1); %Variance is linearly decreasing
end
simDatSetVariableNoise=dset(simDatSetNoiseless.in,simDatSetNoiseless.out+noise);

%% Step 3: identify models
opts.Nreps=1; %Single rep, yes. Based on the fact that the first rep is almost always the definitive one.
opts.fastFlag=200; %Set to 1
opts.indB=1;
opts.indD=[];
warning('off','statKSfast:fewSamples') %This is needed to avoid a warning bombardment
[fitMdl,outlog]=linsys.id(simDatSetFixedNoise,1:6,opts); %Fixed noise only

opts.Nreps=20; %Repeat with 20 reps
warning('off','statKSfast:fewSamples') %This is needed to avoid a warning bombardment
[fitMdl20,outlog20]=linsys.id(simDatSetFixedNoise,1:6,opts); %Fixed noise only


%% Save
save modelOrderTestS1_20Reps.mat fitMdl fitMdl20 outlog outlog20 simDatSetFixedNoise datSet model simDatSetNoiseless stateE

%% Step 5: use fitted models to evaluate log-L and goodness of fit
%%
legacy_vizDataLikelihood(fitMdl,simDatSetFixedNoise) %Fixed noise
set(gcf,'Name','E-M fits to synthetic data from 4th order LTI-SSM (1 rep)')

legacy_vizDataLikelihood(fitMdl20,simDatSetFixedNoise) %Fixed noise
set(gcf,'Name','E-M fits to synthetic data from 4th order LTI-SSM (20 rep)')

for i=1:length(fitMdl)
    stateE=fitMdl{i}.Ksmooth(simDatSetFixedNoise); %SMoothing to get a not-improper prior for init cond, similar to the init estmate that comes from E-M
    initC=stateE.getSample(1);
    l1(i)=fitMdl{i}.logL(simDatSetFixedNoise,initC); %Notice this is not the estimate that E-M comes up with: in EM we are also getting a max likelihood estimate of the initial state
    stateE=fitMdl20{i}.Ksmooth(simDatSetFixedNoise); %SMoothing to get a not-improper prior for init cond, similar to the init estmate that comes from E-M
    initC=stateE.getSample(1);
    l20(i)=fitMdl20{i}.logL(simDatSetFixedNoise,initC);
end
signLogLgap=sign(l20-l1)
log10logLgap=log10(abs(l20-l1))
eig1=cell2mat(cellfun(@(x) sort(eig(x.A)),fitMdl,'UniformOutput',false));
eig20=cell2mat(cellfun(@(x) sort(eig(x.A)),fitMdl20,'UniformOutput',false));
eigDiff=eig1-eig20
%% Test the notion that init covar goes monotonically down and logL up with repeated smoothing of data
initC=initCond([],[]);
i=4; %True model order
logl=[];
tr=[];
for j=1:1000
    j
    [stateE,~,~,~,logl(j)]=fitMdl{i}.Ksmooth(simDatSetFixedNoise,initC); %SMoothing to get a not-improper prior for init cond, similar to the init estmate that comes from E-M
    initC=stateE.getSample(1);
    tr(j)=trace(initC.covar);
end
figure; subplot(2,1,1); plot(tr)
subplot(2,1,2); plot(logl)
