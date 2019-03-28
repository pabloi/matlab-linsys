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

%% Step 2: generate a similar model but with decreasing noise levels
scaleFactor=[ones(1,150) [1500:-1:1]/1500+1];
noise=nan(D2,N);
for i=1:N
    noise(:,i)=sqrt(scaleFactor(i))*cR'*randn(D2,1); %Variance is linearly decreasing
end
simDatSetVariableNoise=dset(simDatSetNoiseless.in,simDatSetNoiseless.out+noise);

%% Step 3: identify models
opts.Nreps=1;
opts.fastFlag=50; 
opts.indB=1;
opts.indD=[];
opts.includeOutputIdx=find(~flatIdx);
warning('off','statKSfast:fewSamples') %This is needed to avoid a warning bombardment

%Fit 1: standard
opts.refineFastFlag=0;
tic
[fitMdlstd,outlogSTD]=linsys.id(simDatSetFixedNoise,4,opts); %Fixed noise only
t1=toc
%Fit 2: enforce fast refining:
opts.refineFastFlag=50;
tic
[fitMdlfastRefine,outlogFR]=linsys.id(simDatSetFixedNoise,4,opts); %Fixed noise only
t2=toc

%% Save
save EMrefine.mat fitMdl* outlog* simDatSetFixedNoise datSet model simDatSetNoiseless stateE t1 t2
%%
fittedLinsys.compare({fitMdlstd;fitMdlfastRefine})
