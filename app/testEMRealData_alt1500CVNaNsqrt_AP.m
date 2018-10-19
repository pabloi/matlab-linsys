%%
addpath(genpath('./'))
%%
clear all
%% Load real data:
[Y,Yf,Ycom,Uf]=groupDataToMatrixForm(true);
Yf=Yf-nanmedian(Yf(1:50,:,:)); %Subtracting Baseline
Yf=nanmedian(Yf,3)'; %Median across subjects, using nanmedian to exploit fast mode (less than 10 missing strides)
Yf=Yf(:,1:1350); %Using only 400 of Post
Uf=Uf(:,1:1350);
%% Split data:
Yff{1}=Yf(:,1:850);
Uff{1}=Uf(1:850);
Yff{2}=Yf(:,851:end);
Uff{2}=Uf(:,851:end);
%
load EMrealDimCompare1500CVsqrtAP.mat
%% Flat model:
[J,B,C,D,Q,R]=getFlatModel(Yff{1},Uff{1}); %Base+Adapt, minus last 100 of Adapt
model{1,1}=autodeal(J,B,C,D,Q,R);
model{1,1}.name='Static, CV1';
[J,B,C,D,Q,R]=getFlatModel(Yff{2},Uff{2});
model{1,2}=autodeal(J,B,C,D,Q,R);
model{1,2}.name='Static, CV2';
%%
for D1=3:4
%% Identify
    tic
    opts.robustFlag=false;
    opts.outlierReject=false;
    opts.fastFlag=false; %Cannot do fast for NaN filled data
    for k=1:2
        [fAh,fBh,fCh,D,fQh,R,fXh,fPh,logL]=randomStartEM(Yff{k},Uff{k},D1,20,opts); %Slow/true EM
        model{D1+1}.runtime=toc;
        [J,B,C,X,~,Q,P] = canonize(fAh,fBh,fCh,fXh,fQh,fPh);
        model{D1+1,k}=autodeal(J,B,C,D,X,Q,R,P,logL);
        model{D1+1,k}.name=['EM (' num2str(D1) ', CV' num2str(k) ')']; %Robust mode does not do fast filtering
    end
end
%%
save EMrealDimCompare1500CVsqrtAP.mat
%% COmpare
%%Train set:
for k=1:2
    vizDataFit(model(1:5,k),Yff{k},Uff{k})
    set(gcf,'Name',['CV' num2str(k) ', training data'])
end
%% Test set:
for k=1:2
    vizDataFit(model(1:5,3-k),Yff{k},Uff{k})
set(gcf,'Name',['CV' num2str(3-k) ', testing data'])
end
%%All:
for k=1:2
    vizDataFit(model(1:5,k),Yf,Uf)
set(gcf,'Name',['CV' num2str(k) ', ALL data'])
end
%% See models
for k=1:2
    vizModels(model(1:5,k))
end
