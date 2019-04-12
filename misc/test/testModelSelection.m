%% Script to test which of the model order selection criteria works best in practice
clear all
close all
addpath(genpath('../../'))
%% Step 1: simulate a high-dim model
load ../../../EMG-LTI-SSM/res/allDataModels.mat
model=model{4}; %3-rd order model as ground truth
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

%% Step 2: identify models
opts.Nreps=10;
opts.fastFlag=200; %Generally a bad idea, but in EM we expect covariance matrices to converge to near steady-state values, so this is fine.
opts.indB=1;
opts.indD=[];
warning('off','statKSfast:fewSamples') %This is needed to avoid a warning bombardment
[fitMdl,outlog]=linsys.id(simDatSetFixedNoise,1:6,opts); %Fixed noise only

%% Save
save testModelSelection.mat fitMdl outlog simDatSetFixedNoise datSet model simDatSetNoiseless stateE

%%
clear all
load('testModelSelection.mat')
fittedLinsys.compare(fitMdl)
set(gcf,'Name','E-M fits to synthetic data from 4th order LTI-SSM (1 reps)')