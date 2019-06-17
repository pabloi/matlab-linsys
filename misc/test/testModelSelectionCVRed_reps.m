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
parpool(4)
M=4;
N=25;
load testModelSelectionCVRed_reps.mat
cv2=cvlogl;
dr2=zeros(M,N,9,maxOrder+1); %NEED TO REPLACE IF RESUMING FROM PREVIOUSLY COMPUTED SOLUTIONS!
cvlogl=zeros(M,N,9,maxOrder+1);
detResiduals=zeros(M,N,9,maxOrder+1);
startPoint=18; %To resume a previous run
cvlogl(:,1:startPoint-1,:,:)=cv2(:,1:startPoint-1,:,:);
detResiduals(:,1:startPoint-1,:,:)=dr2(:,1:startPoint-1,:,:);
%% Extend:

for k=startPoint:N
    k
    parfor reps=1:M  %PARFOR
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
        trainDset=[{simDatSet}; datSetOE; datSetBlk; datSetAP; datSetBlk100];
        testDset=[{''}; datSetOE([2,1]); datSetBlk([2,1]); datSetAP([2,1]); datSetBlk100([2,1])];
        [mdls,outlog{reps,k}]=linsys.id(trainDset,0:maxOrder,opts1);
        %[mdls,outlog{reps,k}]=linsys.id(trainDset(1:2),3,opts1);
        fitMdl{reps,k}=mdls;
        for i=1:9
            for l=1:(maxOrder+1)
                if i>1
                    cvlogl(reps,k,i,l)=mdls{l,i}.logL(testDset{i});
                end
                detResiduals(reps,k,i,l)=sqrt(nansum(nansum(mdls{l,i}.residual(trainDset{i},'det').^2)));
            end
        end
    end
save testModelSelectionCVRed_reps.mat fitMdl outlog opts model cvlogl detResiduals
end
%%
%Notes: this was done enforcing stability, with default minimum values for
%Q,R, single rep (local max?), non-fast flag, subsampling relevant output
%variables.


%% Do some stats on CV and in-sample methods for model selection criterion
load testModelSelectionCVRed_reps.mat
mdl=fitMdl(:); %All models in a unidimensional array
cv=reshape(cvlogl(:,1:size(fitMdl,2),:,:),numel(mdl),size(cvlogl,3),size(cvlogl,4));
%Get summary metrics:
pNames={'goodnessOfFit','BIC','AIC','AICc'};
dSetNames={'all','odd[1]','even[1]','odd[20]','even[20]','first half','second half','odd[100]','even[100]'};
Nparams=length(pNames);
for i=1:length(mdl)
    for k=1:Nparams
        metrics.(pNames{k})(:,:,i)=cell2mat(cellfun(@(x) x.(pNames{k}),mdl{i},'UniformOutput',false));
    end
end

%% Plotting logL, AIC, BIC, AICc, CV logL for all datasets
figure('Units','Pixels','InnerPosition',[100 100 300*3 300*5])   
for i=1:9 %all 9 datasets
    for k=1:Nparams
        subplot(9,5,k+(i-1)*5)
        hold on
         cc=get(gca,'ColorOrder');
        d=squeeze(metrics.(pNames{k})(:,i,:));
        if k==1
            %bestModel=0; %To do
            [~,bestModel]=max(d); %To do
            ylabel(dSetNames{i})
        else
            d=-d/2;
            [~,bestModel]=max(d); 
        end
        d=d-d(1,:);
        plot(0:5,d,'Color',cc(k,:))
        if i==1
        title(pNames{k})
        end
        clear count
        for l=0:5 %For each order, count how many times it is selected by the criterion
            count(l+1)=sum((l+1)==bestModel);
            %text(l,mean(d(l+1,:)),num2str(count),'Color','k','FontSize',12)
        end
        ax=gca;
        ax.YAxis.Limits(1)=0;
        y=ax.YAxis.Limits(2);
        ax.YAxis.TickValues=[];
        ax.XAxis.Limits=[-.5 5.5];
        bb=bar(0:5, y*count/numel(bestModel), 'FaceColor','k');
        uistack(bb,'bottom')
        if i==9
            ax.XAxis.TickValues=[0:5];
            xlabel('Model order')
        else
            ax.XAxis.TickValues=[];
        end
    end
    %Add cv-logl
    if i>1
    subplot(9,5,i*5)
    hold on
    d=squeeze(cv(:,i,:));
    d=d-d(:,1);
    [~,bestModel]=max(d,[],2); 
    plot(0:5,d','Color',cc(5,:))
    ax=gca;
    ax.YAxis.Limits(1)=0;
    if i==2
    title('CV logL')
    end
    clear count
    for l=0:5 %For each order, count how many times it is selected by the criterion
        count(l+1)=sum((l+1)==bestModel);
        %text(l-.3,mean(d(l+1,:)),num2str(count),'Color','r','FontSize',12)
    end
    y=ax.YAxis.Limits(2);
    bb=bar(0:5, y*count/numel(bestModel), 'FaceColor','k');
    uistack(bb,'bottom')
    ax.XAxis.Limits=[-.5 5.5];
    ax.YAxis.TickValues=[];
        if i==9
            ax.XAxis.TickValues=[0:5];
            xlabel('Model order')
        else
            ax.XAxis.TickValues=[];
        end
    end
end

%% A cleaner version: logL (LRT), AIC, BIC, AICc just for the full dataset, CV log-L for half-splits
figure('Units','Pixels','InnerPosition',[100 100 300*3 300*2])   
i=1;
for k=1:Nparams
    subplot(3,4,k+(i-1)*4)
    hold on
     cc=get(gca,'ColorOrder');
    d=squeeze(metrics.(pNames{k})(:,i,:));
    if k==1
        %bestModel=0; %To do
        [~,bestModel]=max(d); %To do
        ylabel(dSetNames{i})
    else
        d=-d/2;
        [~,bestModel]=max(d); 
    end
    d=d-d(1,:);
    plot(0:5,d,'Color',cc(k,:))
    if i==1
    title(pNames{k})
    end
    clear count
    for l=0:5 %For each order, count how many times it is selected by the criterion
        count(l+1)=sum((l+1)==bestModel);
        %text(l,mean(d(l+1,:)),num2str(count),'Color','k','FontSize',12)
    end
    ax=gca;
    ax.YAxis.Limits(1)=0;
    y=ax.YAxis.Limits(2);
    ax.YAxis.TickValues=[];
    ax.XAxis.Limits=[-.5 5.5];
    bb=bar(0:5, y*count/numel(bestModel), 'FaceColor','k');
    uistack(bb,'bottom')
    if i==9
        ax.XAxis.TickValues=[0:5];
        xlabel('Model order')
    else
        ax.XAxis.TickValues=[];
    end
end
%Add cv-logl
cvNames={'Odd/even[1]','Odd/even[20]','First/Second','Odd/even[100]'};
for i=2:9
    subplot(3,4,4+mod(i,2)*4+floor(i/2))
    hold on
    d=squeeze(cv(:,i,:));
    d=d-d(:,1);
    [~,bestModel]=max(d,[],2); 
    plot(0:5,d','Color',cc(5,:))
    ax=gca;
    ax.YAxis.Limits(1)=0;
    if mod(i,2)==0
        title(['CV logL ' cvNames(floor(i/2))])
    end
    clear count
    for l=0:5 %For each order, count how many times it is selected by the criterion
        count(l+1)=sum((l+1)==bestModel);
        %text(l-.3,mean(d(l+1,:)),num2str(count),'Color','r','FontSize',12)
    end
    y=ax.YAxis.Limits(2);
    bb=bar(0:5, y*count/numel(bestModel), 'FaceColor','k');
    uistack(bb,'bottom')
    ax.XAxis.Limits=[-.5 5.5];
    ax.YAxis.TickValues=[];
    if mod(i,2)==1
        ax.XAxis.TickValues=[0:5];
        xlabel('Model order')
    else
        ax.XAxis.TickValues=[];
    end
end
%%
%Get table with summary parameters?