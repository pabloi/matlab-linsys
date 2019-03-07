%% Script to test which of the model order selection criteria works best in practice
addpath(genpath('../../'))
%% Step 1: simulate a high-dim model
load ../../../EMG-LTI-SSM/res/allDataModels.mat
model=model{5}; %4-th order model as ground truth
initC=initCond(zeros(4,1),zeros(4));
deterministicFlag=false;
[simDatSet,stateE]=model.simulate(datSet.in,initC,deterministicFlag);
%% Get folded data for adapt/post
datSetAP=simDatSet.split([751,901]); %First half is 1-750, last part 901-1650
datSetAP=datSetAP([1,3]); %Discarding the middle part
%% Get odd/even data
datSetOE=alternate(simDatSet,2);
%% Step 2: identify models for various orders
opts.Nreps=5;
opts.fastFlag=200;
opts.indB=1;
opts.indD=[];
warning('off','statKSfast:fewSamples') %This is needed to avoid a warning bombardment
[fitMdlAP,outlogAP]=linsys.id(datSetAP,1:6,opts); %Fit A/P models
[fitMdlOE,outlogOE]=linsys.id(datSetOE,1:6,opts); %Fit A/P models

%%
save CVmodelOrderTestS5Reps.mat fitMdlAP fitMdlOE outlog simDatSet datSetAP datSetOE model stateE

%% Step 5: use fitted models to evaluate log-L and goodness of fit
%evaluateModelOrderSelection