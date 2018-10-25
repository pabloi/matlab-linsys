%%
addpath(genpath('./'))
%%
clear all
%% Load real data:
[Y,Yf,Ycom,Uf]=groupDataToMatrixForm(true,true);
Yf=Yf-nanmedian(Yf(1:150,:,:)); %Subtracting Baseline
Yf=nanmedian(Yf,3)'; %Median across subjects, using nanmedian to exploit fast mode (less than 10 missing strides)
%% Flat model:
Yaux=nan(size(Yf));
Yaux(:,[1:2:end])=Yf(:,[1:2:end]);
[J,B,C,D,Q,R]=getFlatModel(Yaux,Uf);
model{1,1}=autodeal(J,B,C,D,Q,R);
model{1,1}.name='Flat, CV1';
Yaux=nan(size(Yf));
Yaux(:,[2:2:end])=Yf(:,[2:2:end]);
[J,B,C,D,Q,R]=getFlatModel(Yaux,Uf);
model{1,2}=autodeal(J,B,C,D,Q,R);
model{1,2}.name='Flat, CV2';
%%
for D1=1:5
%% Identify
    tic
    opts.robustFlag=false;
    opts.outlierReject=false;
    opts.fastFlag=true; %Cannot do fast for NaN filled data
    opts.logFlag=true;
    for k=1:2
        Yaux=nan(size(Yf));
        Yaux(:,[k:2:end])=Yf(:,[k:2:end]); %First 10 strides are given to both sets
        [fAh,fBh,fCh,D,fQh,R,fXh,fPh,logL,outLog]=randomStartEM(Yaux,Uf,D1,20,opts); %Slow/true EM
        model{D1+1}.runtime=toc;
        [J,B,C,X,~,Q,P] = canonize(fAh,fBh,fCh,fXh,fQh,fPh);
        model{D1+1,k}=autodeal(J,B,C,D,X,Q,R,P,logL,outLog);
        model{D1+1,k}.name=['EM (' num2str(D1) ', CV' num2str(k) ')']; %Robust mode does not do fast filtering
    end
end
%%
save ./app/data/EMrealDim_CV2_AP.mat
%% COmpare
%%Train set:
for k=1:2
    Yaux=nan(size(Yf));
    Yaux(:,k:2:end)=Yf(:,k:2:end);
    vizDataFit(model(2:6,k),Yaux,Uf)
    set(gcf,'Name',['CV' num2str(k) ', training data'])
end
%% Test set:
for k=1:2
    Yaux=nan(size(Yf));
    Yaux(:,k:2:end)=Yf(:,k:2:end);
    vizDataFit(model(2:6,3-k),Yaux,Uf)
set(gcf,'Name',['CV' num2str(3-k) ', testing data'])
end
%%All:
for k=1:2
    vizDataFit(model(2:5,k),Yf,Uf)
    set(gcf,'Name',['CV' num2str(k) ', ALL data'])
end
%% See models
for k=1:2
    vizModels(model(2:6,k))
end
