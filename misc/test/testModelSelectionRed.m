%% Script to test which of the model order selection criteria works best in practice
clear all
close all
addpath(genpath('../../'))
%% Step 1: simulate a high-dim model
load ../../../EMG-LTI-SSM/res/allDataModels.mat
model=model{4}; %3rd order model as ground truth
initC=initCond(zeros(3,1),zeros(3));
deterministicFlag=false;

%%
opts.Nreps=5; %Single rep, yes. Based on the fact that the first rep is almost always the definitive one.
opts.fastFlag=50; %Set to 1
opts.indB=1;
opts.indD=[];
opts.stableA=true;
opts.refineTol=1e-3; 
opts.refineMaxIter=2e3;

 warning('off','statKSfast:fewSamples') %This is needed to avoid a warning bombardment
parfor rep=1:100
    rep
    simDatSetFixedNoise=model.simulate(datSet.in,initC);

    % Reduce data
    Y=simDatSetFixedNoise.out;
    U=simDatSetFixedNoise.in;
    X=Y-(Y/U)*U; %Projection over input
    s=var(X'); %Estimate of variance
    flatIdx=s<.005; %Variables are split roughly in half at this threshold
    opts1=opts;
    opts1.includeOutputIdx=find(~flatIdx);

    % identify models
    [fitMdlRed(:,rep),outlog{rep}]=linsys.id(simDatSetFixedNoise,0:5,opts1); %Fixed noise only
end

%% Save
save testModelSelectionRed_100reps.mat fitMdlRed outlog model
%%
load testModelSelectionRed_100reps.mat
%Plot sample rep:
fittedLinsys.compare(fitMdlRed{1})
%Get stats from all reps:

%Plot summary:
