%%
addpath(genpath('../aux/'))
addpath(genpath('../kalman/'))
addpath(genpath('../data/'))
addpath(genpath('../EM/'))
addpath(genpath('../sPCA/'))
addpath(genpath('../../robustCov/'))
%%
clear all
%% Load real data:
[Y,Yf,Ycom,Uf]=groupDataToMatrixForm();
Yf=Yf-nanmedian(Yf(1:50,:,:)); %Subtracting Baseline
Yf=median(Yf,3)'; %Median across subjects
Yf=Yf(:,1:1350); %Using only 400 of Post
Uf=Uf(:,1:1350);
%% Flat model:
[J,B,C,D,Q,R]=getFlatModel(Yf(:,1:2:end),Uf(:,1:2:end));
model{1,1}=autodeal(J,B,C,D,Q,R);
model{1,1}.name='Flat, CV1';
[J,B,C,D,Q,R]=getFlatModel(Yf(:,2:2:end),Uf(:,2:2:end));
model{1,2}=autodeal(J,B,C,D,Q,R);
model{1,2}.name='Flat, CV2';
%%
for D1=1:5
%% Identify
    tic
    opts.robustFlag=false;
    opts.outlierReject=false;
    opts.fastFlag=false; %Cannot do fast for NaN filled data
    for k=1:2
        Yaux=nan(size(Yf));
        Yaux(:,k:2:end)=Yf(:,k:2:end);
        [fAh,fBh,fCh,D,fQh,R,fXh,fPh,logL]=randomStartEM(Yaux,Uf,D1,20,opts); %Slow/true EM
        model{D1+1}.runtime=toc;
        [J,B,C,X,~,Q,P] = canonizev2(fAh,fBh,fCh,fXh,fQh,fPh);
        model{D1+1,k}=autodeal(J,B,C,D,X,Q,R,P,logL);
        model{D1+1,k}.name=['EM (' num2str(D1) ', CV' num2str(k) ')']; %Robust mode does not do fast filtering
    end
end
%%
save EMrealDimCompare1500CV2.mat
%% COmpare
%%Train set:
for k=1:2
vizDataFit(model(2:5,k),Yf(:,k:2:end),Uf(:,k:2:end))
end
%% Test set:
for k=1:2
vizDataFit(model(2:5,3-k),Yf(:,k:2:end),Uf(:,k:2:end))
end
