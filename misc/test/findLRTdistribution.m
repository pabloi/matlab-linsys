%% Script to test which of the model order selection criteria works best in practice
addpath(genpath('../../'))
%% Step 1: simulate a high-dim model
load ../../../EMG-LTI-SSM/res/allDataModels.mat
model=model{2}; %1-st order model as ground truth
initC=initCond(zeros(4,1),zeros(4));
deterministicFlag=false;


%% Step 2: identify models for various orders
opts.Nreps=0; %Very fast estimation
reps=1e2;
simDatSet=cell(reps,1);
maxOrder=6;
fitMdl=cell(reps,maxOrder);
outlog=cell(reps,maxOrder);
for i=1:reps
    %Simulate realization:
    [simDatSet{i},stateE]=model.simulate(datSet.in,initC,deterministicFlag);
    %Get MLE for various orders:
    [fitMdl(i,:),outlog(i,:)]=linsys.id(simDatSet{i},1:maxOrder,opts);
end
save LRTdistr_trueOrder1_SingleRep.mat fitMdl outlog simDatSet datSet model initC