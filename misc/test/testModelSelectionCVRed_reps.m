%% Script to test which of the model order selection criteria works best in practice
addpath(genpath('../../'))
clear all
%% Step 1: simulate a high-dim model
%load ../../../EMG-LTI-SSM/res/allDataModels.mat
load ../../../EMG-LTI-SSM/res/allDataRedAlt_20190510T175706.mat %Different model from the one above
model=modelRed{4}; %3rd order model as ground truth
%Replacing R diagonal elements with proper variances (reduced model did not
%estimate those variances):
infIdx=isinf(diag(model.R));
vR=var(datSet.out-model.D*datSet.in,[],2);
vR(~infIdx)=0;
model.R(isinf(model.R))=0; %Setting to 0 first
model.R=model.R+diag(vR);
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
%parpool(4)
M=4;
N=25;
startPoint=2; %To resume a previous run
dr2=zeros(M,N,9,maxOrder+1); %NEED TO REPLACE IF RESUMING FROM PREVIOUSLY COMPUTED SOLUTIONS!
cvlogl=zeros(M,N,9,maxOrder+1);
detResiduals=zeros(M,N,9,maxOrder+1);
if startPoint>1
    load testModelSelectionCVRed_repsAlt.mat
    cv2=cvlogl;
    cvlogl(:,1:startPoint-1,:,:)=cv2(:,1:startPoint-1,:,:);
    detResiduals(:,1:startPoint-1,:,:)=dr2(:,1:startPoint-1,:,:);
end



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
save testModelSelectionCVRed_repsAlt.mat fitMdl outlog opts model cvlogl detResiduals
end
%%
%Notes: this was done enforcing stability, with default minimum values for
%Q,R, single rep (local max?), non-fast flag, subsampling relevant output
%variables.


%% Do some stats on CV and in-sample methods for model selection criterion
clear all
load testModelSelectionCVRed_repsAlt.mat
mdl=fitMdl(:); %All models in a unidimensional array
cv=reshape(cvlogl(:,1:size(fitMdl,2),:,:),numel(mdl),size(cvlogl,3),size(cvlogl,4));
dr=reshape(detResiduals(:,1:size(fitMdl,2),:,:),numel(mdl),size(cvlogl,3),size(cvlogl,4));
%Get summary metrics:
pNames={'fittedLogL','BIC','AIC','AICc'};
dSetNames={'all','odd[1]','even[1]','odd[20]','even[20]','first half','second half','odd[100]','even[100]'};
Nparams=length(pNames);
for i=1:length(mdl)
    for k=1:Nparams
        metrics.(pNames{k})(:,:,i)=cell2mat(cellfun(@(x) x.(pNames{k}),mdl{i},'UniformOutput',false));
    end
end

%% Another metric for CV:
%for each of the 100 models fit (to all data or half data, does not
%matter), we can compute how they cross-validate to an altogether new
%realization of the same model.
%This cannot be used for model selection in practice, but can serve to indicate what the expected level of generalization of the models is (which is the ultimate question).
%e.g. AIC/AICc should roughly be an unbiased estimator of CV log-L, but is
%it?
for j=1:100 %Generate 100 new realizations
for i=1:length(mdl)
    %mdl{i}.logL() %To do
end
end

%% Also, we can estimate parameter variability across fitted models:
%Reminder: this results are likely VERY dependent on proper refinement of
%solution. To avoid long computation times, the refinement stage for this
%analysis is looser than the EM defaults on matlab-linsys.
for i=1:length(mdl)
    aux=mdl{i};
    for ord=1:6        
    taus{ord}(i,:)=sort(-1./log(eig(aux{ord,1}.A)));
    end
end
%Reminder: the true taus where 53,106,1374
median(taus{4})


%% Plotting logL, AIC, BIC, AICc, CV logL for all datasets
figure('Units','Pixels','InnerPosition',[100 100 300*3 300*5])   
for i=1:9 %all 9 datasets
    for k=1:Nparams
        subplot(9,6,k+(i-1)*6)
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
    subplot(9,6,(i-1)*6+5)
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
    %Add det residuals:
    subplot(9,6,(i-1)*6+6)
    hold on
    d=squeeze(dr(:,i,:));
    d=d(:,1)-d;
    d=d(69:end,:); %First 68 reps did not compute residuals
    [~,bestModel]=max(d,[],2); 
    plot(0:5,d','Color',cc(6,:))
    ax=gca;
    ax.YAxis.Limits(1)=0;
    if i==1
    title('1-det. residuals')
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

%% A cleaner version: logL (LRT), AIC, BIC, AICc just for the full dataset, CV log-L for half-splits
f=figure('Units','Pixels','InnerPosition',[100 100 300*3 300*1.5])   
i=1;
M=numel(fitMdl);
for k=1:Nparams
    %subplot(3,4,k+(i-1)*4)
    axes('Position',[.08+(k-1)*.22 .58 .2 .37])
    hold on
     cc=get(gca,'ColorOrder');
    d=squeeze(metrics.(pNames{k})(:,i,:));
    if k==1
        %bestModel=0; %To do
        %[~,bestModel]=max(d); %Maximum likelihood
        clear bestModel
        deltaL=diff(d,[],1)';
        dof=cellfun(@(x) x.dof,fitMdl{1,1}(:,1))';
        th=chi2inv(.95,diff(dof));
        signifDiff=2*deltaL>th;
        for kk=1:M
        aux=find([1 double(signifDiff(kk,:))]==0,1,'first')-1; %First failed test
        if isempty(aux) %All tests passed
           bestModel(kk)=length(th)+1; 
        else
            bestModel(kk)=aux;
        end
        end
        ylabel(dSetNames{i})
    else
        d=-d/2;
        [~,bestModel]=max(d); 
    end
    d=d-d(1,:);
    plot(0:5,d,'Color',cc(k,:))
    if i==1
        if k==1
            title('LRT')
        else
            title(pNames{k})
        end
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
    bb=bar(0:5, y*count/numel(bestModel), 'FaceColor','k','EdgeColor','none');
    uistack(bb,'bottom')
    if i==9
        ax.XAxis.TickValues=[0:5];
        xlabel('Model order')
    else
        ax.XAxis.TickValues=[];
    end
    ax.YAxis.Label.String='';
    ax.YAxis.TickValues=[.1 .3 .5 .7 .9]*y;
    if k==1
    ax.YAxis.TickLabels={'10%','','50%','','90%'};
    else
    ax.YAxis.TickLabels={};    
    end
    axes(ax)
        ax.YGrid='on';
            text(2.6,y*count(4)/numel(bestModel)+.04*y,[num2str(100*count(4)/numel(bestModel),2) '%'],'FontSize',10,'Color','k')
end
%Add cv-logl
cvNames={'blocks [1]','blocks [20]','first/second halves','blocks [100]'};
% for i=2:9
%     %subplot(3,4,4+mod(i,2)*4+floor(i/2))
%     axes('Position',[.08+(floor(i/2)-1)*.22 .36-mod(i,2)*.25 .2 .22])
%     hold on
%     d=squeeze(cv(:,i,:));
%     d=d-d(:,1);
%     [~,bestModel]=max(d,[],2); 
%     plot(0:5,d','Color',cc(5,:))
%     ax=gca;
%     ax.YAxis.Limits(1)=0;
%     if mod(i,2)==0
%         title(['CV log-L ' cvNames(floor(i/2))])
%     end
%     clear count
%     for l=0:5 %For each order, count how many times it is selected by the criterion
%         count(l+1)=sum((l+1)==bestModel);
%         %text(l-.3,mean(d(l+1,:)),num2str(count),'Color','r','FontSize',12)
%     end
%     y=ax.YAxis.Limits(2);
%     bb=bar(0:5, y*count/numel(bestModel), 'FaceColor','k');
%     uistack(bb,'bottom')
%     ax.XAxis.Limits=[-.5 5.5];
%     ax.YAxis.TickValues=[];
%     ax.XAxis.TickValues=[0:5];
%     if mod(i,2)==1
%         xlabel('Model order')
%     else
%         ax.XAxis.TickLabels={};
%     end
%     ax.YAxis.TickValues=[.1 .3 .5 .7 .9]*y;
%     if i<4
%     ax.YAxis.TickLabels={'10%','','50%','','90%'};
%     else
%     ax.YAxis.TickLabels={};    
%     end
%     ax.YGrid='on';
% end
%ALTERNATE VISUALIZATION:
aux=[2,4,8,6]; %Re-ordering for easier presentation
for j=1:4
    i=aux(j);
    %subplot(3,4,4+mod(i,2)*4+floor(i/2))
    axes('Position',[.08+(j-1)*.22 .1 .2 .37])
    hold on
    d=squeeze(mean(cv(:,[i,i+1],:),2));
    d=d-d(:,1);
    [~,bestModel]=max(d,[],2); 
    plot(0:5,d','Color',cc(5,:))
    ax=gca;
    ax.YAxis.Limits(1)=0;
    if mod(i,2)==0
        title(['CV log-L ' cvNames(floor(i/2))])
    end
    clear count
    for l=0:5 %For each order, count how many times it is selected by the criterion
        count(l+1)=sum((l+1)==bestModel);
        %text(l-.3,mean(d(l+1,:)),num2str(count),'Color','r','FontSize',12)
    end
    y=ax.YAxis.Limits(2);
    bb=bar(0:5, y*count/numel(bestModel), 'FaceColor','k','EdgeColor','none');
    uistack(bb,'bottom')
    ax.XAxis.Limits=[-.5 5.5];
    ax.YAxis.TickValues=[];
    ax.XAxis.TickValues=[0:5];
    if mod(i,2)==0
        xlabel('Model order')
    else
        ax.XAxis.TickLabels={};
    end
    ax.YAxis.TickValues=[.1 .3 .5 .7 .9]*y;
    if i<4
    ax.YAxis.TickLabels={'10%','','50%','','90%'};
    else
    ax.YAxis.TickLabels={};    
    end
    ax.YGrid='on';
    text(2.6,y*count(4)/numel(bestModel)+.04*y,[num2str(100*count(4)/numel(bestModel),2) '%'],'FontSize',10,'Color','k')
end
% Average all cv results and see what results:
%cvl=reshape(mean(cvlogl(:,:,[2,3,4,5,8,9],:),3),100,6); %Ignoring the first/second half split
%[~,idx]=max(cvl,[],2);
%sum(idx==4)/sum(idx~=1) %If this is below the percentage of correct choices 
%for the 100 stride blocked result, then averaging across data splits does not help. 
%The interpretation in that case are that CV log-L methods all fail in
%essentially the same situations, when there is an 'unlucky' data draw
tt=findobj(gcf,'Type','Text');
set(tt,'FontName','OpenSans');
pp=findobj(gcf,'Type','Axes');
set(pp,'FontName','OpenSans');

%%
addpath('../../../ext/altmany-export_fig-b1a7288/')
%export_fig repeatedModelSelectionResults.eps -eps -c[0 5 0 5] -transparent -m2 -r600 -depsc
export_fig repeatedModelSelectionResultsAlt.png -png -c[0 5 0 5] -transparent -r300
%%
%Get table with summary parameters?









