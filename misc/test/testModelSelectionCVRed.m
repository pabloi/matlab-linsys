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
%%
Y=simDatSet.out;
U=simDatSet.in;
X=Y-(Y/U)*U; %Projection over input
s=var(X'); %Estimate of variance
flatIdx=s<.005; %Variables are split roughly in half at this threshold
%% Step 2: identify models for various orders
opts.Nreps=20; %Single rep, this works well enough for full (non-CV) data
opts.fastFlag=0; %Should not use fast for O/E because of NaN
opts.indB=1;
opts.indD=[];
opts.stableA=true;
opts.fastFlag=100;
opts.includeOutputIdx=find(~flatIdx);
numCores = feature('numcores');
%p = parpool(numCores);

%Run identification with all data, true order, as reference
mdlAll=linsys.id(simDatSet,3,opts);

%Cross-validation first/second halves
[fitMdlAPRed,outlogAP]=linsys.id([datSetAP],0:6,opts);

%Cross-validation alternating:
opts.fastFlag=false;
[fitMdl,outlog]=linsys.id([datSetOE; datSetBlk],0:6,opts);

%Separate models:
fitMdlBlkRed=fitMdl(:,3:4);
fitMdlOERed=fitMdl(:,1:2);


%%
save testModelSelectionCVRed.mat fitMdlAPRed fitMdlOERed fitMdlBlkRed outlogAP outlog simDatSet datSetAP datSetOE datSetBlk model stateE opts mdlAll

%% Step 5: use fitted models to evaluate log-L and goodness of fit
load testModelSelectionCVRed.mat

%CV log-l:
f1=figure;
[fh] = vizCVDataLikelihood(fitMdlAPRed(:,:),datSetAP([2,1]));
fh.Name='Early/late CV';
%ph=findobj(fh,'Type','Axes');
%p1=copyobj(ph,f1);
%p1.Position
fh=vizCVDataLikelihood(fitMdlOERed,datSetOE([2,1]));
fh.Name='Odd/even CV';
fh=vizCVDataLikelihood(fitMdlBlkRed,datSetBlk([2,1]));
fh.Name='Blocked (20) CV';

%Train set log-L:
fittedLinsys.compare(fitMdlAPRed(:,1))
fittedLinsys.compare(fitMdlAPRed(:,2))
fittedLinsys.compare(fitMdlOERed(:,1))
fittedLinsys.compare(fitMdlOERed(:,2))
%Need to put these 4 in a single figure, set proper size, remove unnecessary text, export as
%eps
fittedLinsys.compare(fitMdlBlkRed(:,1))
fittedLinsys.compare(fitMdlBlkRed(:,2))
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

%% Show results as table:
clear all
load testModelSelectionCVRed.mat
aux=model.R;
model.R=Inf(size(aux));
model.R(opts.includeOutputIdx,opts.includeOutputIdx)=aux(opts.includeOutputIdx,opts.includeOutputIdx);
mdl={model, mdlAll, fitMdlBlkRed{4,1}, fitMdlBlkRed{4,2},fitMdlOERed{4,1}, fitMdlOERed{4,2},fitMdlAPRed{4,1}, fitMdlAPRed{4,2}};
mdl{1}.name='True';
mdl{3}.name='Odd blocks';
mdl{4}.name='Even blocks';
mdl{7}.name='First half';
mdl{8}.name='Second half';
mdl{5}.name='Odd samples';
mdl{6}.name='Even samples';
mdl{2}.name='All data';
linsys.summaryTable(mdl)