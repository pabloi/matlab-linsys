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
opts.Nreps=10;
opts.fastFlag=1;
opts.indB=1;
opts.indD=[];
warning('off','statKSfast:fewSamples') %This is needed to avoid a warning bombardment
%[fitMdl,outlog]=linsys.id({simDatSetFixedNoise},1:6,opts); %Fixed noise only
[fitMdl,outlog]=linsys.id({simDatSetFixedNoise; simDatSetVariableNoise},1:6,opts);
fitMdlFixedNoise=fitMdl(:,1);
outlogFixedNoise=outlog(:,1);
fitMdlVariableNoise=fitMdl(:,2);
outlogVariableNoise=outlog(:,2);

%% Save
save modelOrderTestS10RepsWVariableNoise.mat fitMdlFixedNoise fitMdlVariableNoise outlogFixedNoise outlogVariableNoise simDatSetFixedNoise simDatSetVariableNoise datSet model simDatSetNoiseless stateE

%% Step 5: use fitted models to evaluate log-L and goodness of fit
%%
legacy_vizDataLikelihood(fitMdlFixedNoise,simDatSetFixedNoise) %Fixed noise
set(gcf,'Name','E-M fits to synthetic data from 4th order LTI-SSM')

%%
legacy_vizDataLikelihood(fitMdlVariableNoise,simDatSetVariableNoise) %Variable noise
set(gcf,'Name','E-M fits to synthetic data from 4th order LTI-SSM, with variable R')