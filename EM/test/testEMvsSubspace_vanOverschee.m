addpath(genpath('../../'))
clear all
%% Step 1: get a high-dim model

%Data-inspired:
load ../../../EMG-LTI-SSM/res/allDataModels.mat
in=datSet.in;
model=model{4}; %2nd order model as ground truth

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
Nx=model.order;
initC=initCond(zeros(Nx,1),zeros(Nx));

%% Simulate different data from the model, and identify, given that we KNOW the true model order
Nreps=100;
opts.Nreps=1; %Single rep, this works well enough
opts.fastFlag=100; %So it ends fast
opts.indB=1;
opts.indD=[];
opts.stableA=true;
opts.Niter=200; %So it ends fast
opts.refineTol=.005; %So it ends fast
opts.refineFastFlag=false; %Set to false to disable fast refine, generally just makes the slow refine MUCH slower
opts.disableRefine=true;
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

save EMvsSubspace_vanOverschee200iters_noRefineOrder3.mat model EMid SSid opts initC deterministicFlag in tEM tSS
    
%% Compare log-likelihoods and runtime
%load EMvsSubspace_vanOverschee200iters_noRefineOrder3.mat
%load EMvsSubspace_vanOverschee100iters_noRefineOrder3.mat
load EMvsSubspace_vanOverschee100iters_noRefine.mat %Fast version
%load EMvsSubspace_vanOverschee.mat %Slow version
f=figure('Units','Pixels','InnerPosition',[100 100 300*3 300]);
N=model.order;
ph=subplot(1,3,1);
%bar([mean(tEM) mean(tSS) mean(tSS2)])
hold on
bar(1,mean(tEM),'EdgeColor','none','FaceAlpha',.7)
bar(2,mean(tSS),'EdgeColor','none','FaceAlpha',.7)
title('Algorithm runtime (s)')
%ylabel('Mean time (s)')
set(gca,'XTick',[1:2],'XTickLabel',{'EM','SS'},'YScale','log','YTick',[mean(tSS) mean(tEM)])
bb=axis;
ax=gca;
ax.YAxis.Limits=[1 100];

subplot(1,3,2)
%plot([1 2 3],[cellfun(@(x) x.goodnessOfFit,EMid)' cellfun(@(x) x.goodnessOfFit,SSid)' cellfun(@(x) x.goodnessOfFit,SSid2)'],'kx')
boxplot([cellfun(@(x) x.goodnessOfFit,EMid)' cellfun(@(x) x.goodnessOfFit,SSid)'])
set(gca,'XTick',[1,2],'XTickLabel',{'EM','SS'})
%ylabel('log-L')
aa=axis;
axis([bb(1:2) aa(3:4)])
title('log-likelihood')

subplot(1,3,3)
hold on;
trueTau=-1./log(eig(model.A));
for j=1:2
    switch j
        case 1
            tbl=linsys.summaryTable(EMid); %EM results
        case 2
            tbl=linsys.summaryTable(SSid); %SS results

    end
    matrix=tbl.Variables;
    taus=matrix(:,[1:3:3*N]);
    taus(taus<0)=4950;%exp(10);
    taus(taus>=4900)=exp(9); %4900 is the largest allowed in EM, so it means saturation effectively
    taus=sort(taus,2);
    u=taus./trueTau';
    %mean(u)
    %std(u)
    histogram(log(real(taus(:))),3.5:.1:exp(9.5),'EdgeColor','none');
end
%legend('EM \tau', 'SS \tau', 'SSEM \tau')
ax=gca;
lg=legend('EM','SS');
lg.AutoUpdate='off';
plot(log(sort(trueTau')).*[1;1],80*[0;1]','k--')
xlabel('log(\tau_i)')
ylabel('Count')
if N==2
%For the two order one:
set(gca,'XTick',[3,log(trueTau(1)),5,6,7,log(trueTau(2)),9],'XTickLabel',{'3','\tau_1','5','6','7','\tau_2','>8.5'})
axis([3 9.5 0 65])  
elseif N==3
%For the three order:
set(gca,'XTick',[3,log(trueTau(1)),4,log(trueTau(2)),5,6,7,log(trueTau(3)),9],'XTickLabel',{'3','\tau_1','4','\tau_2','5','6','7','\tau_3','>8.5'})
axis([3 9.5 0 65])  
end
title('Time-constants identified')

    
    
    