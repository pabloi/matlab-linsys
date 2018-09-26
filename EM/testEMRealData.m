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
for i=1:3 %B,A,P
    %Remove baseline
    data{i}=allDataEMG{i}-B;

    %Interpolate over NaNs
    for j=1:size(data{i},3) %each subj
        t=1:size(data{i},1); nanidx=any(isnan(data{i}(:,:,j)),2); %Any muscle missing
        data{i}(:,:,j)=interp1(t(~nanidx),data{i}(~nanidx,:,j),t,'linear','extrap'); %Substitute nans
    end
    
    %Remove subj:
    data{i}=data{i}(:,:,subjIdx);
    
    %Compute asymmetry component
    aux=data{i}-fftshift(data{i},2);
    dataSym{i}=aux(:,1:size(aux,2)/2,:);
    
    
end

%% All data
Y=[median(dataSym{1},3); median(dataSym{2},3);median(dataSym{3},3)]';
U=[zeros(size(dataSym{1},1),1);ones(size(dataSym{2},1),1);zeros(size(dataSym{3},1),1);]';
%% Just B and A
Y=[median(dataSym{1},3); median(dataSym{2},3)]';
U=[zeros(size(dataSym{1},1),1);ones(size(dataSym{2},1),1)]';
%% Median-filtered B, A
binw=3;
Y2=[medfilt1(median(dataSym{1},3),binw,'truncate'); medfilt1(median(dataSym{2},3),binw,'truncate')]';
%%
D1=4;
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
tic
opts.Niter=500;
opts.robustFlag=true;
opts.fastFlag=[];
opts.outlierReject=true;
% [fAh,fBh,fCh,D,fQh,R,fXh,fPh]=EM(Y2,U,model{1}.X,opts); %Median filtered data
[fAh,fBh,fCh,D,fQh,R,fXh,fPh]=EM(Y,U,model{1}.X,opts); 
logL=dataLogLikelihood(Y,U,fAh,fBh,fCh,D,fQh,R,fXh(:,1),fPh(:,:,1));
model{2}.runtime=toc
[J,B,C,X,~,Q,P] = canonizev2(fAh,fBh,fCh,fXh,fQh,fPh);
model{2}=autodeal(J,B,C,D,X,Q,R,P,logL);
model{2}.name='EM (robust, outlier reject,smooth start)';

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

%%
save EMreal4.mat
%% COmpare
vizModels(model)
vizDataFit(model,Y,U)
