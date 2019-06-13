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

%Blocked CV with size 100
datSetBlk100=simDatSet.blockSplit(100,2); 
[fitMdlBlkRed100,outlogBlkRed100]=linsys.id(datSetBlk100,0:6,opts);


%%
save testModelSelectionCVRed.mat fitMdlAPRed fitMdlOERed fitMdlBlkRed fitMdlBlkRed100 outlogAP outlog outlogBlkRed100 simDatSet datSetAP datSetOE datSetBlk datSetBlk100 model stateE opts mdlAll

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
fh=vizCVDataLikelihood(fitMdlBlkRed100,datSetBlk100([2,1]));
fh.Name='Blocked (100) CV';
%To do: put all in single figure, make pretty

%% in-sample criteria:
load testModelSelectionRed.mat
fitMdlRed=[{linsys.id(simDatSet,0,opts)};fitMdlRed];
mdlList=[fitMdlRed fitMdlAPRed fitMdlOERed fitMdlBlkRed fitMdlBlkRed100];
name={'All','First-half','Second-half','Odd','Even','Odd blks (20)','Even blks (20)','Odd blks (100)','Even blks (100)'};
f1=figure('Units','Pixels','InnerPosition',[100 100 300*2 300*4]);
M=size(mdlList,2);
for i=1:M
    fh=fittedLinsys.compare(mdlList(:,i));
    ph=findobj(fh,'Type','Axes');
    p1=copyobj(ph,f1);
            close(fh)
    if i==1
        set(p1,'XTickLabel',{'Flat','1','2','3','4','5','6'})
    else
        set(p1,'XTickLabel',{})
    end
    p1=p1(end:-1:1);
    p1(1).YAxis.Label.String=name{i};
    delete(p1(end))
    for k=1:length(p1)-1
       p1(k).Position=[.1+(k-1)*.225 .05+(i-1)*(.93/M) .2 .9*.93/M]; 
       if i~=M
       p1(k).Title.String='';
       end
       if k~=1
           p1(k).YTickLabel={};
       end
       axes(p1(k))
       grid off
       bb=findobj(p1(k),'Type','bar');
       set(bb,'EdgeColor','w','FaceAlpha',.5);
       tt=findobj(p1(k),'Type','text');
       set(tt,'Color','k')
       if k==1
       for kk=1:length(tt)
          tt(kk).String=[regexp(tt(kk).String,'p.*$','match')];%regexp('p*',tt(kk).String) ;
       end
       end
    end
end
saveFig(f1,'./','inSampleModelSelection',0)

%% Show results as table:
clear all
load testModelSelectionCVRed.mat
aux=model.R;
model.R=Inf(size(aux));
model.R(opts.includeOutputIdx,opts.includeOutputIdx)=aux(opts.includeOutputIdx,opts.includeOutputIdx);
mdl={model, mdlAll, fitMdlBlkRed{4,1}, fitMdlBlkRed{4,2},fitMdlBlkRed100{4,1}, fitMdlBlkRed100{4,2},fitMdlOERed{4,1}, fitMdlOERed{4,2},fitMdlAPRed{4,1}, fitMdlAPRed{4,2}};
mdl{1}.name='True';
mdl{3}.name='Odd blocks 20';
mdl{4}.name='Even blocks 20';
mdl{5}.name='Odd blocks 100';
mdl{6}.name='Even blocks 100';
mdl{9}.name='First half';
mdl{10}.name='Second half';
mdl{7}.name='Odd samples';
mdl{8}.name='Even samples';
mdl{2}.name='All data';
linsys.summaryTable(mdl)