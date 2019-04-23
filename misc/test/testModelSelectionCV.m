%% Script to test which of the model order selection criteria works best in practice
addpath(genpath('../../'))
clear all
%% Step 1: simulate a high-dim model
load ../../../EMG-LTI-SSM/res/allDataModels.mat
model=model{4}; %3rd order model as ground truth
initC=initCond(zeros(3,1),zeros(3));
deterministicFlag=false;
[simDatSet,stateE]=model.simulate(datSet.in,initC,deterministicFlag);
clear datSet
%% Get folded data for adapt/post
datSetAP=simDatSet.split([826]); %Split in half
%% Get odd/even data
datSetOE=alternate(simDatSet,2);
%% Get blocked data
blkSize=20; %This discards the last 10 samples, leaves the first 10 (exactly) after each transition on a different set
datSetBlk=simDatSet.blockSplit(blkSize,2); 
%% Step 2: identify models for various orders
opts.Nreps=10; %Single rep, this works well enough for full (non-CV) data
opts.fastFlag=0; %Should not use fast for O/E because of NaN
opts.indB=1;
opts.indD=[];
opts.stableA=true;
opts.fastFlag=100;
numCores = feature('numcores');
p = parpool(numCores);
[fitMdlAP,outlogAP]=linsys.id([datSetAP],0:6,opts);
opts.fastFlag=false;
[fitMdl,outlog]=linsys.id([datSetOE; datSetBlk],0:6,opts);

%Separate models:
fitMdlBlk=fitMdl(:,3:4);
fitMdlOE=fitMdl(:,1:2);

%%
save testModelSelectionCV.mat fitMdlAP fitMdlOE fitMdlBlk outlogAP outlog simDatSet datSetAP datSetOE datSetBlk model stateE

%% Step 5: use fitted models to evaluate log-L and goodness of fit
load testModelSelectionCV.mat

%CV log-l:
[fh] = vizCVDataLikelihood(fitMdlAP(1:4,:),datSetAP([2,1]));
fh.Name='Early/late CV';
fh=vizCVDataLikelihood(fitMdlOE,datSetOE([2,1]));
fh.Name='Odd/even CV';
fh=vizCVDataLikelihood(fitMdlBlk,datSetBlk([2,1]));
fh.Name='Blocked (20) CV';

%Train set log-L:
fittedLinsys.compare(fitMdlAP(:,1))
fittedLinsys.compare(fitMdlAP(:,2))
fittedLinsys.compare(fitMdlOE(:,1))
fittedLinsys.compare(fitMdlOE(:,2))
fittedLinsys.compare(fitMdlBlk(:,1))
fittedLinsys.compare(fitMdlBlk(:,2))
%[fh] = vizCVDataLikelihood(fitMdlAP,datSetAP);

%vizCVDataLikelihood(fitMdlBlk,datSetBlk);
% ah=copyobj(ph([2,3]),fh);
% ah(1).Title.String={'Adapt-model';'Cross-validation'};
% ah(1).YAxis.Label.String={'Post-data'; 'log-L'};
% ah(2).Title.String={'Adapt-model';'-BIC/2'};
% ah(2).XTickLabel={'1','2','3','4','5','6'};
% ah(1).XTickLabel={'1','2','3','4','5','6'};
% ah1=copyobj(ph1([1,4]),fh);
% ah1(2).Title.String={'Post-model';'Cross-validation'};
% ah1(2).YAxis.Label.String={'Adapt-data';'log-L'};
% ah1(2).XAxis.Label.String={'Model Order'};
% ah1(2).XTickLabel={'1','2','3','4','5','6'};
% ah1(1).XAxis.Label.String={'Model Order'};
% ah1(1).XTickLabel={'1','2','3','4','5','6'};
% ah1(1).Title.String={'Post-model';'-BIC/2'};
% set(gcf,'Name','Adapt/Post cross-validation');
