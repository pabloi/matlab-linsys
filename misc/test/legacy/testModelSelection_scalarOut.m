%% Script to test which of the model order selection criteria works best in practice
addpath(genpath('../../'))
%% Step 1: simulate a high-dim model
load ../../../EMG-LTI-SSM/res/allDataModels.mat
model=model{5}; %4-th order model as ground truth
initC=initCond(zeros(4,1),eye(4));
deterministicFlag=false;
model.C=model.C(1,:);
model.D=model.D(1,:);
model.R=.0001*model.R(1,1);
[simDatSet,stateE]=model.simulate(datSet.in,initC);

%% Step 2: identify models for various orders
opts.Nreps=5;
opts.fastFlag=0;
opts.indB=1;
opts.indD=[];
opts.robustFlag=false;
warning('off','statKSfast:fewSamples') %This is needed to avoid a warning bombardment
[fitMdl,outlog]=linsys.id(simDatSet,1:6,opts);

save modelOrderTestS5Reps_scalarOut.mat fitMdl outlog simDatSet datSet model stateE

%% Step 3: use fitted models to evaluate log-L and goodness of fit
legacy_vizDataLikelihood(fitMdl,simDatSet) %Fixed noise
set(gcf,'Name','E-M fits to synthetic data from 4th order LTI-SSM, scalar out')
