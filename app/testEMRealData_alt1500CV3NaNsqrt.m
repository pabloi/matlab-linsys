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
%% Flat model:
[J,B,C,D,Q,R]=getFlatModel(Yf(:,1:2:end),Uf(:,1:2:end));
model{1,1}=autodeal(J,B,C,D,Q,R);
model{1,1}.name='Flat, CV1';
[J,B,C,D,Q,R]=getFlatModel(Yf(:,2:2:end),Uf(:,2:2:end));
model{1,2}=autodeal(J,B,C,D,Q,R);
model{1,2}.name='Flat, CV2';
%%
for D1=1:4
%% Identify
    tic
    opts.robustFlag=false;
    opts.outlierReject=false;
    opts.fastFlag=false; %Cannot do fast for NaN filled data
    for k=1:3
        Yaux=nan(size(Yf));
        Yaux(:,[k:3:end])=Yf(:,[k:3:end]);
        [fAh,fBh,fCh,D,fQh,R,fXh,fPh,logL]=randomStartEM(Yaux,Uf,D1,20,opts); %Slow/true EM
        model{D1+1}.runtime=toc;
        [J,B,C,X,~,Q,P] = canonize(fAh,fBh,fCh,fXh,fQh,fPh);
        model{D1+1,k}=autodeal(J,B,C,D,X,Q,R,P,logL);
        model{D1+1,k}.name=['EM (' num2str(D1) ', CV' num2str(k) ')']; %Robust mode does not do fast filtering
    end
end
%%
save EMrealDimCompare1500CV3sqrt.mat
%% COmpare
%%Train set:
for k=1:3
    Yaux=nan(size(Yf));
    Yaux(:,k:3:end)=Yf(:,k:3:end);
    vizDataFit(model(2:5,k),Yaux,Uf)
    set(gcf,'Name',['CV' num2str(k) ', training data'])
end
%% Test set:
for k=1:3
    Yaux=Yf;
    Yaux(:,k:3:end)=NaN;
    vizDataFit(model(2:5,4-k),Yaux,Uf)
set(gcf,'Name',['CV' num2str(4-k) ', testing data'])
end
%%All:
for k=1:3
    vizDataFit(model(2:5,k),Yf,Uf)
    set(gcf,'Name',['CV' num2str(k) ', ALL data'])
end
%% See models
for k=1:3
    vizModels(model(2:5,k))
    set(gcf,'Name',['CV' num2str(k) ', model viz'])
end
