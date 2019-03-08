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
opts.Nreps=10;
opts.fastFlag=1;
opts.indB=1;
opts.indD=[];
warning('off','statKSfast:fewSamples') %This is needed to avoid a warning bombardment
[fitMdl,outlog]=linsys.id([datSetAP;datSetOE],1:6,opts); %Fit all models jointly, to better exploit parallelism
%[fitMdl,outlog]=linsys.id([datSetAP],1:6,opts);
%Separate models:
fitMdlAP=fitMdl(:,1:2);
fitMdlOE=fitMdl(:,3:4);
outlogAP=fitMdl(:,1:2);
outlogOE=fitMdl(:,3:4);

%%
save CVmodelOrderTestS10Reps.mat fitMdlAP fitMdlOE outlogOE outlogAP simDatSet datSetAP datSetOE model stateE

%% Step 5: use fitted models to evaluate log-L and goodness of fit
%Odd-trained on even data
f1=vizDataLikelihood(fitMdlAP(:,1),datSetAP);
ph=findobj(gcf,'Type','Axes');
f2=vizDataLikelihood(fitMdlAP(:,2),datSetAP);
ph1=findobj(gcf,'Type','Axes');

fh=figure;
ah=copyobj(ph([2,3]),fh);
ah(1).Title.String={'Adapt-model';'Cross-validation'};
ah(1).YAxis.Label.String={'Post-data'; 'log-L'};
ah(2).Title.String={'Adapt-model';'-BIC/2'};
ah(2).XTickLabel={'1','2','3','4','5','6'};
ah(1).XTickLabel={'1','2','3','4','5','6'};
ah1=copyobj(ph1([1,4]),fh);
ah1(2).Title.String={'Post-model';'Cross-validation'};
ah1(2).YAxis.Label.String={'Adapt-data';'log-L'};
ah1(2).XAxis.Label.String={'Model Order'};
ah1(2).XTickLabel={'1','2','3','4','5','6'};
ah1(1).XAxis.Label.String={'Model Order'};
ah1(1).XTickLabel={'1','2','3','4','5','6'};
ah1(1).Title.String={'Post-model';'-BIC/2'};
set(gcf,'Name','Adapt/Post cross-validation');
%close(f1)
%close(f2)

%%
f1=vizDataLikelihood(fitMdlOE(:,1),datSetOE);
ph=findobj(gcf,'Type','Axes');
f2=vizDataLikelihood(fitMdlOE(:,2),datSetOE);
ph1=findobj(gcf,'Type','Axes');

fh=figure;
ah=copyobj(ph([2,3]),fh);
ah(1).Title.String={'Odd-model';'Cross-validation'};
ah(1).YAxis.Label.String={'Even-data'; 'log-L'};
ah(2).Title.String={'Odd-model';'-BIC/2'};
ah(2).XTickLabel={'1','2','3','4','5','6'};
ah(1).XTickLabel={'1','2','3','4','5','6'};
ah1=copyobj(ph1([1,4]),fh);
ah1(2).Title.String={'Even-model';'Cross-validation'};
ah1(2).YAxis.Label.String={'Odd-data';'log-L'};
ah1(2).XAxis.Label.String={'Model Order'};
ah1(2).XTickLabel={'1','2','3','4','5','6'};
ah1(1).XAxis.Label.String={'Model Order'};
ah1(1).XTickLabel={'1','2','3','4','5','6'};
ah1(1).Title.String={'Even-model';'-BIC/2'};
set(gcf,'Name','Odd/even cross-validation');
close(f1)
close(f2)

