addpath(genpath('../../'))
clear all
%% Step 1: get a high-dim model

%Data-inspired:
load ../../../EMG-LTI-SSM/res/allDataModels.mat
in=datSet.in;
model=model{3}; %2nd order model as ground truth
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
ssSize=40;
for i=1:Nreps
    i
    deterministicFlag=false;
    [simDatSet,stateE]=model.simulate(in,initC,deterministicFlag);
    tic
    EMid{i}=linsys.id(simDatSet,2,opts);
    tEM(i)=toc;
    EMid{i}.name=['EM_' num2str(i)];
    tic
    SSid{i}=linsys.SSid(simDatSet,2,ssSize);
    tSS(i)=toc;
    SSid{i}.name=['SS_' num2str(i)];
    tic
    SSid2{i}=linsys.SSEMid(simDatSet,2,10); %This does not appear to benefit from using large ssSize
    tSS2(i)=toc;
    SSid2{i}.name=['SS2_' num2str(i)];
    %data=iddata(simDatSet.out',simDatSet.in');
    %mod=n4sid(data,2,'Feedthrough',true); %Very slow for large dimensional data
    %N4Sid{i}=fittedLinsys(mod.A,mod.C,eye(size(model.R)),mod.B,mod.D,mod.K*mod.K',initC,simDatSet,['N4S' num2str(i)],[],NaN,[]);
end

save EMvsSubspace_i40.mat model EMid SSid SSid2 opts initC deterministicFlag in tEM tSS tSS2 ssSize
%% Make table and compare time-constants:
    figure; hold on;
for j=1:3
    switch j
        case 1
            tbl=linsys.summaryTable(EMid);
        case 2
            tbl=linsys.summaryTable(SSid);
        case 3
            tbl=linsys.summaryTable(SSid2);
    end
    matrix=tbl.Variables;
    taus=matrix(:,[1,3]);
    taus(taus<0)=4950;%exp(10);
    histogram(log(real(taus(:))),3:.1:10,'EdgeColor','none');
end
legend('EM \tau', 'SS \tau', 'SSEM \tau')
    plot(log(sort(-1./log(eig(model.A))))'.*[1;1],80*[0;1]','k--')
    
%% Compare log-likelihoods and runtime
figure;
subplot(2,1,1)
bar([mean(tEM) mean(tSS) mean(tSS2)])
title('Mean run time (s)')

subplot(2,1,2)
plot([1 2 3],[cellfun(@(x) x.goodnessOfFit,EMid)' cellfun(@(x) x.goodnessOfFit,SSid)' cellfun(@(x) x.goodnessOfFit,SSid2)'],'kx')
set(gca,'XTick',[1,2,3],'XTickLabel',{'EM','SS','SSEM'})
title('log-L')