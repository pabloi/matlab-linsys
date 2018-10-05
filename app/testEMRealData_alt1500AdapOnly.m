%%
addpath(genpath('../aux/'))
addpath(genpath('../kalman/'))
addpath(genpath('../data/'))
addpath(genpath('../sPCA/'))
addpath(genpath('../EM/'))
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
%% Just B and A
Y=Yf(:,1:950);
U=Uf(:,1:950);
%% Just P
Yp=Yf(:,951:end);
Up=Uf(:,951:end);
%% P and some A
Y_p=Yf(:,850:end);
U_p=Uf(:,850:end);
%% Flat model:
[J,B,C,D,Q,R]=getFlatModel(Y,U);
model{1}=autodeal(J,B,C,D,Q,R);
model{1}.name='Flat';
%%
for D1=1:4
%% Identify
    tic
    opts.robustFlag=false;
    opts.Niter=1500;
    opts.outlierReject=false;
    opts.fastFlag=true;
    [fAh,fBh,fCh,D,fQh,R,fXh,fPh]=randomStartEM(Y,U,D1,10,opts); %Slow/true EM
    logL=dataLogLikelihood(Y,U,fAh,fBh,fCh,D,fQh,R,fXh(:,1),fPh(:,:,1));
    model{D1+1}.runtime=toc;
    [J,B,C,X,~,Q,P] = canonizev2(fAh,fBh,fCh,fXh,fQh,fPh);
    model{D1+1}=autodeal(J,B,C,D,X,Q,R,P,logL);
    model{D1+1}.name=['EM (iterated,all,' num2str(D1) ')']; %Robust mode does not do fast filtering
end
%%
save EMrealDimCompare1500_Aonly.mat
%% COmpare
vizModels(model(1:4))
%%
%vizDataFit(model([4:-1:1]),Y,U)
%vizDataFit(model([4:-1:1]),Y_p,U_p)
vizDataFit(model(1:4),Yf,Uf)
