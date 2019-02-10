%% Script to test which of the model order selection criteria works best in practice
addpath(genpath('../../'))
%% Step 1: simulate a high-dim model
load ../../../EMG-LTI-SSM/res/allDataModels.mat
model=model{5}; %4-th order model as ground truth
initC=initCond(zeros(4,1),zeros(4));
deterministicFlag=false;
[simDatSet,stateE]=model.simulate(datSet.in,initC,deterministicFlag);

%% Step 2: identify models for various orders
opts.Nreps=5;
for order=1:6
    [fitMdl{order},outlog{order}]=linsys.id(simDatSet,order,opts);
end
save modelOrderTestS5Reps.mat fitMdl outlog simDatSet datSet model
%% Step 3: use fitted models to evaluate log-L and goodness of fit
vizDataLikelihood(fitMdl,simDatSet)