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

%% Reduce data
Y=simDatSetFixedNoise.out;
U=simDatSetFixedNoise.in;
X=Y-(Y/U)*U; %Projection over input
s=var(X'); %Estimate of variance
flatIdx=s<.005; %Variables are split roughly in half at this threshold


%% Step 3: identify models
opts.Nreps=5; %Single rep, yes. Based on the fact that the first rep is almost always the definitive one.
opts.fastFlag=50; %Set to 1
opts.indB=1;
opts.indD=[];
opts.includeOutputIdx=find(~flatIdx);
warning('off','statKSfast:fewSamples') %This is needed to avoid a warning bombardment
[fitMdlRed,outlog]=linsys.id(simDatSetFixedNoise,1:6,opts); %Fixed noise only

%% Save
save testModelSelectionRed.mat fitMdlRed outlog simDatSetFixedNoise datSet model simDatSetNoiseless stateE
%%
load testModelSelectionRed.mat
fittedLinsys.compare(fitMdlRed)