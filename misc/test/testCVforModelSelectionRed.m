%% Script to test which of the model order selection criteria works best in practice
addpath(genpath('../../'))
%% Step 1: simulate a high-dim model
load ../../../EMG-LTI-SSM/res/allDataModels.mat
model=model{5}; %4-th order model as ground truth
initC=initCond(zeros(4,1),zeros(4));
deterministicFlag=false;
[simDatSet,stateE]=model.simulate(datSet.in,initC,deterministicFlag);
%% Reduce data
Y=datSet.out;
U=datSet.in;
X=Y-(Y/U)*U; %Projection over input
s=var(X'); %Estimate of variance
flatIdx=s<.005; %Variables are split roughly in half at this threshold
%% Get folded data for adapt/post
datSetAP=simDatSet.split([826]); %Split in half
%% Get odd/even data
datSetOE=alternate(simDatSet,2);
%% Step 2: identify models for various orders
opts.Nreps=10; %Single rep, this works well enough for full (non-CV) data
opts.fastFlag=0; %Should not use fast for O/E because of NaN
opts.indB=1;
opts.indD=[];
opts.includeOutputIdx=find(~flatIdx); 
warning('off','statKSfast:fewSamples') %This is needed to avoid a warning bombardment
[fitMdl,outlog]=linsys.id([datSetOE; datSetAP],1:6,opts);

%Separate models:
fitMdlAP=fitMdl(:,3:4);
fitMdlOE=fitMdl(:,1:2);
outlogAP=outlog(:,3:4);
outlogOE=outlog(:,1:2);

%%
save CVmodelOrderTestS10RepsRed.mat fitMdlAP fitMdlOE outlogOE outlogAP simDatSet datSetAP datSetOE model stateE

%% Step 5: use fitted models to evaluate log-L and goodness of fit
%Odd-trained on even data
f1=vizDataLikelihood(fitMdlAP(:,1),datSetAP);
ph=findobj(gcf,'Type','Axes');
f2=vizDataLikelihood(fitMdlAP(:,2),datSetAP);
ph1=findobj(gcf,'Type','Axes');

fh=figure;
ah=copyobj(ph([1]),fh);
ah(1).Title.String={'Adapt-model';'Cross-validation'};
ah(1).YAxis.Label.String={'Post-data'; '\Delta log-L'};
ah(1).XTickLabel={'1','2','3','4','5','6'};
ah1=copyobj(ph1([2]),fh);
ah1(1).Title.String={'Post-model';'Cross-validation'};
ah1(1).XAxis.Label.String={'Model Order'};
ah1(1).XTickLabel={'1','2','3','4','5','6'};
ah1(1).YAxis.Label.String={'Adapt-data'; '\Delta log-L'};
set(gcf,'Name','Adapt/post cross-validation');
%close(f1)
%close(f2)

%%
f1=vizDataLikelihood(fitMdlOE(:,1),datSetOE);
ph=findobj(gcf,'Type','Axes');
f2=vizDataLikelihood(fitMdlOE(:,2),datSetOE);
ph1=findobj(gcf,'Type','Axes');

fh=figure;
ah=copyobj(ph([1]),fh);
ah(1).Title.String={'Odd-model';'Cross-validation'};
ah(1).YAxis.Label.String={'Even-data'; '\Delta log-L'};
ah(1).XTickLabel={'1','2','3','4','5','6'};
ah1=copyobj(ph1([2]),fh);
ah1(1).Title.String={'Even-model';'Cross-validation'};
ah1(1).XAxis.Label.String={'Model Order'};
ah1(1).XTickLabel={'1','2','3','4','5','6'};
ah1(1).YAxis.Label.String={'Odd-data'; '\Delta log-L'};
set(gcf,'Name','Odd/even cross-validation');

