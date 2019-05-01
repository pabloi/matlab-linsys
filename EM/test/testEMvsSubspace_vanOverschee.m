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
    SSid{i}=linsys.SSid(simDatSet,2,4,'subid'); %Using van Overschee's implementation, which is the fastest one
    tSS(i)=toc;
    SSid{i}.name=['SS4_' num2str(i)];
end

save EMvsSubspace_vanOverschee.mat model EMid SSid opts initC deterministicFlag in tEM tSS
    
%% Compare log-likelihoods and runtime
figure;

ph=subplot(3,1,1);
%bar([mean(tEM) mean(tSS) mean(tSS2)])
hold on
bar(1,mean(tEM),'EdgeColor','none')
bar(2,mean(tSS),'EdgeColor','none')
title('Algorithm runtime')
ylabel('Mean time (s)')
set(gca,'XTick',[1:6],'XTickLabel',{'EM','SS4'})
bb=axis;

subplot(3,1,2)
%plot([1 2 3],[cellfun(@(x) x.goodnessOfFit,EMid)' cellfun(@(x) x.goodnessOfFit,SSid)' cellfun(@(x) x.goodnessOfFit,SSid2)'],'kx')
boxplot([cellfun(@(x) x.goodnessOfFit,EMid)' cellfun(@(x) x.goodnessOfFit,SSid)'])
set(gca,'XTick',[126],'XTickLabel',{'EM','SS4'})
ylabel('log-L')
aa=axis;
axis([bb(1:2) aa(3:4)])
title('Solution goodness-of-fit')

subplot(3,1,3)
hold on;
trueTau=-1./log(eig(model.A));
for j=1:2
    switch j
        case 1
            tbl=linsys.summaryTable(EMid);
        case 2
            tbl=linsys.summaryTable(SSid);

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
    
    
    
    
    