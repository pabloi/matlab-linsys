%% Script to test which of the model order selection criteria works best in practice
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
simDatSetFixedNoise=simDatSetNoiseless;
simDatSetFixedNoise=dset(simDatSetNoiseless.in,simDatSetNoiseless.out+noise);

%% Step 2: identify models for various orders
opts.Nreps=5;
opts.fastFlag=200;
opts.indB=1;
opts.indD=[];
warning('off','statKSfast:fewSamples') %This is needed to avoid a warning bombardment
[fitMdl,outlog]=linsys.id(simDatSetFixedNoise,1:6,opts);

%% Step 3: generate a similar model but with decreasing noise levels
scaleFactor=[ones(1,150) [1500:-1:1]/1500+1];
noise=nan(D2,N);
for i=1:N
    noise(:,i)=sqrt(scaleFactor(i))*cR'*randn(D2,1); %Variance is linearly decreasing
end
simDatSetVariableNoise=dset(simDatSetNoiseless.in,simDatSetNoiseless.out+noise);

%% Step 4: fit the new data
opts.Nreps=5;
opts.fastFlag=200;
opts.indB=1;
opts.indD=[];
[fitMdlVariableNoise,outlogVariableNoise]=linsys.id(simDatSetVariableNoise,1:6,opts);

save modelOrderTestS5RepsWVariableNoise.mat fitMdl fitMdlVariableNoise outlog outlogVariableNoise simDatSetFixedNoise simDatSetVariableNoise datSet model simDatSetNoiseless stateE

%% Step 5: use fitted models to evaluate log-L and goodness of fit
evaluateModelOrderSelection