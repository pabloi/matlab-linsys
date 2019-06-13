%% Script to test which of the model order selection criteria works best in practice
addpath(genpath('../../'))
clear all
%% Step 1: simulate a high-dim model
load ../../../EMG-LTI-SSM/res/allDataModels.mat
model=model{4}; %3rd order model as ground truth
initC=initCond(zeros(3,1),zeros(3));
deterministicFlag=false;
maxOrder=5;
%%
opts.Nreps=3;
opts.indB=1;
opts.indD=[];
opts.stableA=true;
opts.fastFlag=false;
opts.refineTol=1e-3; 
opts.refineMaxIter=2e3;
opts.Niter=1e3;
%%
M=7;
for k=1:5 %Do 35 reps, double loop to save partial results
    k
    parfor reps=1:M
        reps
        [simDatSet,stateE]=model.simulate(datSet.in,initC,deterministicFlag);
        %% Get folded data for adapt/post
        datSetAP=simDatSet.split([826]); %Split in half
        %% Get odd/even data
        datSetOE=alternate(simDatSet,2);
        %% Get blocked data
        blkSize=20; %This discards the last 10 samples, leaves the first 10 (exactly) after each transition on a different set
        datSetBlk=simDatSet.blockSplit(blkSize,2); 
        datSetBlk100=simDatSet.blockSplit(100,2); 
        %%
        Y=simDatSet.out;
        U=simDatSet.in;
        X=Y-(Y/U)*U; %Projection over input
        s=var(X'); %Estimate of variance
        flatIdx=s<.005; %Variables are split roughly in half at this threshold
        %% Step 2: identify models for various orders
        opts1=opts;
        opts1.includeOutputIdx=find(~flatIdx);

        %Cross-validation alternating:
        [fitMdl{reps,k},outlog{reps,k}]=linsys.id([{simDatSet}; datSetOE; datSetBlk; datSetAP; datSetBlk100],0:maxOrder,opts1);
    end
save testModelSelectionCVRed_reps.mat fitMdl outlog opts model
end
%%
%Notes: this was done enforcing stability, with default minimum values for
%Q,R, single rep (local max?), non-fast flag, subsampling relevant output
%variables.


%% Do some stats on CV and in-sample methods for model selection criterion
%Plot sample results

%Plot model order choice summary

%Get table with summary parameters?