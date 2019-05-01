addpath(genpath('../../'))
clear all
%% Step 1: get a high-dim model

%Data-inspired:
load ../../../EMG-LTI-SSM/res/allDataModels.mat
in=datSet.in;
model=model{3}; %2nd order model as ground truth

%Removing the constant input: (SS does not play well with it)
model.D=model.D(:,1);
model.B=model.B(:,1);
in=in(1,:);
%model.R=1e-4*model.R; %Very low noise
clear datSet

%Synthetic:
%cR=randn(180);
%cQ=randn(2);
%model=linsys(diag([.98,.995]),randn(180,2),(cR*cR'),.01*[1 0; 1 0],randn(180,2),cQ*cQ');
%in=[zeros(1,150), ones(1,900), zeros(1,600);ones(1,1650)];

initC=initCond(zeros(2,1),zeros(2));

%% Simulate different data from the model, and identify, given that we KNOW the true model order
Nreps=100;
opts.Nreps=1; %Single rep, this works well enough
opts.fastFlag=100; %So it ends fast
opts.indB=1;
opts.indD=[];
opts.stableA=true;
opts.Niter=300; %So it ends fast
opts.refineTol=.005; %So it ends fast
opts.refineFastFlag=true; %Set to false to disable fast refine, generally just makes the slow refine MUCH slower
deterministicFlag=false;
parfor i=1:Nreps
    i
    [simDatSet,stateE]=model.simulate(in,initC,deterministicFlag);
    tic
    EMid{i}=linsys.id(simDatSet,2,opts);
    tEM(i)=toc;
    EMid{i}.name=['EM_' num2str(i)];
    tic
    SSid40{i}=linsys.SSid(simDatSet,2,40);
    tSS40(i)=toc;
    SSid40{i}.name=['SS40_' num2str(i)];
    tic
    SSid10{i}=linsys.SSid(simDatSet,2,10);
    tSS10(i)=toc;
    SSid10{i}.name=['SS10_' num2str(i)];
    tic
    SSidUnb10{i}=linsys.SSidUnb(simDatSet,2,10);
    tSSUnb10(i)=toc;
    SSidUnb10{i}.name=['SSunb_' num2str(i)];
    tic
    SSid20{i}=linsys.SSid(simDatSet,2,20);
    tSS20(i)=toc;
    SSid20{i}.name=['SS20_' num2str(i)];
    tic
    SSEMid{i}=linsys.SSEMid(simDatSet,2,10); %This does not appear to benefit from using large ssSize
    tSSEM(i)=toc;
    SSEMid{i}.name=['SSEM_' num2str(i)];
    %data=iddata(simDatSet.out',simDatSet.in');
    %mod=n4sid(data,2,'Feedthrough',true); %Very slow for large dimensional data
    %N4Sid{i}=fittedLinsys(mod.A,mod.C,eye(size(model.R)),mod.B,mod.D,mod.K*mod.K',initC,simDatSet,['N4S' num2str(i)],[],NaN,[]);
end

save EMvsSubspace_vanOverschee.mat model EMid SSid10 SSid20 SSid40 SSEMid SSidUnb10 opts initC deterministicFlag in tEM tSS20 tSS40 tSS10 tSSEM tSSUnb10
    
%% Compare log-likelihoods and runtime
figure;

ph=subplot(3,1,1);
%bar([mean(tEM) mean(tSS) mean(tSS2)])
hold on
bar(1,mean(tEM),'EdgeColor','none')
bar(2,mean(tSS10),'EdgeColor','none')
bar(3,mean(tSSEM),'EdgeColor','none')
bar(4,mean(tSS20),'EdgeColor','none')
bar(5,mean(tSS40),'EdgeColor','none')
bar(6,mean(tSSUnb10),'EdgeColor','none')
title('Algorithm runtime')
ylabel('Mean time (s)')
set(gca,'XTick',[1:6],'XTickLabel',{'EM','SS10','SSEM','SS20','SS40','SSunb'})
bb=axis;

subplot(3,1,2)
%plot([1 2 3],[cellfun(@(x) x.goodnessOfFit,EMid)' cellfun(@(x) x.goodnessOfFit,SSid)' cellfun(@(x) x.goodnessOfFit,SSid2)'],'kx')
boxplot([cellfun(@(x) x.goodnessOfFit,EMid)' cellfun(@(x) x.goodnessOfFit,SSid10)' cellfun(@(x) x.goodnessOfFit,SSEMid)' cellfun(@(x) x.goodnessOfFit,SSid20)' cellfun(@(x) x.goodnessOfFit,SSid40)' cellfun(@(x) x.goodnessOfFit,SSidUnb10)'])
set(gca,'XTick',[1:6],'XTickLabel',{'EM','SS10','SSEM','SS20','SS40','SSunb'})
ylabel('log-L')
aa=axis;
axis([bb(1:2) aa(3:4)])
title('Solution goodness-of-fit')

subplot(3,1,3)
hold on;
trueTau=-1./log(eig(model.A));
for j=1:6
    switch j
        case 1
            tbl=linsys.summaryTable(EMid);
        case 2
            tbl=linsys.summaryTable(SSid10);
        case 3
            tbl=linsys.summaryTable(SSEMid);
        case 4
            tbl=linsys.summaryTable(SSid20);
        case 5
            tbl=linsys.summaryTable(SSid40);
        case 6
            tbl=linsys.summaryTable(SSidUnb10);
    end
    matrix=tbl.Variables;
    taus=matrix(:,[1,3]);
    taus(taus<0)=4950;%exp(10);
    taus(taus>4950)=4950;
    taus=sort(taus,2);
    u=taus./trueTau';
    mean(u)
    std(u)
    histogram(log(real(taus(:))),3.5:.13:log(7000),'EdgeColor','none');
end
%legend('EM \tau', 'SS \tau', 'SSEM \tau')
plot(log(sort(trueTau')).*[1;1],80*[0;1]','k--')
xlabel('log(\tau_i)')
ylabel('Count')
set(gca,'XTick',[3,4,log(trueTau(1)),5,6,7,log(trueTau(2)),8,8.6],'XTickLabel',{'3','4','\tau_1','5','6','7','\tau_2','8','>8.5'})
axis([3.5 9.5 0 85])  
title('Time-constants identified (histograms)')
    
    
    
    
    