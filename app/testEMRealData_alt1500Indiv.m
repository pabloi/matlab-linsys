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
subj=2;
Yf=[dataSym{1}(:,:,subj); dataSym{2}(:,:,subj);dataSym{3}(:,:,subj)]';
Uf=[zeros(size(dataSym{1}(:,:,subj),1),1);ones(size(dataSym{2}(:,:,subj),1),1);zeros(size(dataSym{3}(:,:,subj),1),1);]';

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
%% Flat model:
model{1}.J=0;
model{1}.B=0;
model{1}.C=ones(size(Y,1),1);
model{1}.D=mean(Yf(:,51:950),2);
model{1}.Q=0;
model{1}.R=eye(size(Y,1));
model{1}.name='Flat';
%%
for D1=1:5
%% Identify
    tic
    opts.robustFlag=false;
    opts.Niter=1500;
    opts.outlierReject=false;
    opts.fastFlag=true;
    [fAh,fBh,fCh,D,fQh,R,fXh,fPh]=randomStartEM(Yf,Uf,D1,10,opts); %Slow/true EM
    logL=dataLogLikelihood(Y,U,fAh,fBh,fCh,D,fQh,R,fXh(:,1),fPh(:,:,1));
    model{D1+1}.runtime=toc;
    [J,B,C,X,~,Q,P] = canonizev2(fAh,fBh,fCh,fXh,fQh,fPh);
    model{D1+1}=autodeal(J,B,C,D,X,Q,R,P,logL);
    model{D1+1}.name=['EM (iterated,all,' num2str(D1) ')']; %Robust mode does not do fast filtering
end
%%
save EMrealDimCompare1500_Subj2.mat
%% COmpare
vizModels(model(1:4))
%%
%vizDataFit(model([4:-1:1]),Y,U)
%vizDataFit(model([4:-1:1]),Y_p,U_p)
vizDataFit(model(1:4),Yf,Uf)