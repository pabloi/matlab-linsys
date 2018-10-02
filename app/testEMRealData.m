%%
addpath(genpath('../aux/')) 
addpath(genpath('../kalman/')) 
addpath(genpath('../data/'))
addpath(genpath('../sPCA/'))
addpath(genpath('../../robustCov/')) 
%%
clear all
%% Load real data:
load ../data/dynamicsData.mat
addpath(genpath('./fun/'))
% Some pre-proc
B=nanmean(allDataEMG{1}(end-45:end-5,:,:)); %Baseline: last 40, exempting 5
clear data dataSym
subjIdx=2:16;
%muscPhaseIdx=[1:(180-24),(180-11:180)];
%muscPhaseIdx=[muscPhaseIdx,muscPhaseIdx+180]; %Excluding PER
muscPhaseIdx=1:360;
for i=1:3 %B,A,P
    %Remove baseline
    data{i}=allDataEMG{i}-B;

    %Interpolate over NaNs %This is only needed if we want to run fast
    %estimations, or if we want to avoid all subjects' data at one
    %timepoint from being discarded because of a single subject's missing
    %data
    for j=1:size(data{i},3) %each subj
       t=1:size(data{i},1); nanidx=any(isnan(data{i}(:,:,j)),2); %Any muscle missing
       data{i}(:,:,j)=interp1(t(~nanidx),data{i}(~nanidx,:,j),t,'linear',0); %Substitute nans
    end
    
    %Two subjects have less than 600 Post strides: C06, C08
    %Option 1: fill with zeros (current)
    %Option 2: remove them
    %Option 3: Use only 400 strides of POST, as those are common to all
    %subjects
    
    %Remove subj:
    data{i}=data{i}(:,muscPhaseIdx,subjIdx);
    
    %Compute asymmetry component
    aux=data{i}-fftshift(data{i},2);
    dataSym{i}=aux(:,1:size(aux,2)/2,:);
    
end

%% All data
Yf=[median(dataSym{1},3); median(dataSym{2},3);median(dataSym{3},3)]';
Uf=[zeros(size(dataSym{1},1),1);ones(size(dataSym{2},1),1);zeros(size(dataSym{3},1),1);]';
Yf=Yf(:,1:1350); %Using only 400 of Post
Uf=Uf(:,1:1350);
%% All data, mean
Ym=[mean(dataSym{1},3); mean(dataSym{2},3);mean(dataSym{3},3)]';
%% Just B and A
Y=Yf(:,1:950);
U=Uf(:,1:950);
%% Just P
Yp=Yf(:,951:end);
Up=Uf(:,951:end);
%% P and some A
Y_p=Yf(:,850:end);
U_p=Uf(:,850:end);
%% Median-filtered B, A
binw=3;
Y2=[medfilt1(median(dataSym{1},3),binw,'truncate'); medfilt1(median(dataSym{2},3),binw,'truncate')]';
%%
D1=3;
%% Identify 1: handcrafted sPCA
tic
model{1}= sPCAv8(Y(:,51:950)',D1,[],[],[]);
model{1}.runtime=toc;
model{1}.X=[zeros(size(model{1}.X,1),size(dataSym{1},1)) model{1}.X];
aux=Y-model{1}.C*model{1}.X-model{1}.D*U;
R=aux*aux'/size(aux,2);
model{1}.R=R+1e-8*eye(size(R));
model{1}.Q=zeros(D1);
logL=dataLogLikelihood(Y,U,model{1}.J,model{1}.B,model{1}.C,model{1}.D,model{1}.Q,model{1}.R,model{1}.X(:,1),model{1}.Q);
[J,B,C,X,~,Q] = canonizev2(model{1}.J,model{1}.B,model{1}.C,model{1}.X,model{1}.Q);
D=model{1}.D;
model{1}=autodeal(J,B,C,D,X,Q,R,logL);
model{1}.name='sPCA';
%% Identify 2: true EM with smooth start, median filtered data
% tic
% opts.Niter=500;
% opts.robustFlag=true;
% opts.fastFlag=[];
% opts.outlierReject=false;
% % [fAh,fBh,fCh,D,fQh,R,fXh,fPh]=EM(Y2,U,model{1}.X,opts); %Median filtered data
% [fAh,fBh,fCh,D,fQh,R,fXh,fPh]=EM(Y,U,model{1}.X,opts); 
% logL=dataLogLikelihood(Y,U,fAh,fBh,fCh,D,fQh,R,fXh(:,1),fPh(:,:,1));
% model{2}.runtime=toc
% [J,B,C,X,~,Q,P] = canonizev2(fAh,fBh,fCh,fXh,fQh,fPh);
% model{2}=autodeal(J,B,C,D,X,Q,R,P,logL);
% model{2}.name='EM (robust,smooth start)';

tic
opts.robustFlag=true;
opts.Niter=500;
opts.outlierReject=false;
opts.fastFlag=true;
[fAh,fBh,fCh,D,fQh,R,fXh,fPh]=randomStartEM(Yf,Uf,D1,10,opts); %Slow/true EM
logL=dataLogLikelihood(Y,U,fAh,fBh,fCh,D,fQh,R,fXh(:,1),fPh(:,:,1));
model{2}.runtime=toc
[J,B,C,X,~,Q,P] = canonizev2(fAh,fBh,fCh,fXh,fQh,fPh);
model{2}=autodeal(J,B,C,D,X,Q,R,P,logL);
model{2}.name='EM (iterated,all)'; %Robust mode does not do fast filtering
%% Identify 3: random start EM, fast
tic
opts.robustFlag=false;
opts.Niter=500;
opts.outlierReject=false;
opts.fastFlag=true;
[fAh,fBh,fCh,D,fQh,R,fXh,fPh]=randomStartEM(Y,U,D1,10,opts); %Slow/true EM
logL=dataLogLikelihood(Y,U,fAh,fBh,fCh,D,fQh,R,fXh(:,1),fPh(:,:,1));
model{3}.runtime=toc
[J,B,C,X,~,Q,P] = canonizev2(fAh,fBh,fCh,fXh,fQh,fPh);
model{3}=autodeal(J,B,C,D,X,Q,R,P,logL);
model{3}.name='EM (iterated)'; %Robust mode does not do fast filtering
%% Identify 4: random start EM, fast, post-data
tic
opts.robustFlag=false;
opts.Niter=500;
opts.outlierReject=false;
opts.fastFlag=true;
[fAh,fBh,fCh,D,fQh,R,fXh,fPh]=randomStartEM(Y_p,U_p,D1,10,opts); %Slow/true EM
logL=dataLogLikelihood(Y,U,fAh,fBh,fCh,D,fQh,R,fXh(:,1),fPh(:,:,1));
model{4}.runtime=toc
[J,B,C,X,~,Q,P] = canonizev2(fAh,fBh,fCh,fXh,fQh,fPh);
model{4}=autodeal(J,B,C,D,X,Q,R,P,logL);
model{4}.name='EM (iterated,post)'; %Robust mode does not do fast filtering
%% Identify 5: flat model
model{5}=model{1};
model{5}.name='flat';
model{5}.J=zeros(D1);
model{5}.B=zeros(size(model{5}.B));
model{5}.D=mean(Yf(:,51:950),2);
%%
save EMreal3_cut.mat
%% COmpare
vizModels(model(2:5))
%%
%vizDataFit(model([4:-1:1]),Y,U)
%vizDataFit(model([4:-1:1]),Y_p,U_p)
vizDataFit(model([5:-1:2]),Yf,Uf)