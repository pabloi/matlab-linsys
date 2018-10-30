%%
addpath(genpath('./'))
%%
clear all
%% Load real data:
[Y,Yf,Ycom,Uf]=groupDataToMatrixForm(true,true);
%NO baseline subtraction
Yf=nanmedian(Yf,3)'; %Median across subjects, using nanmedian to exploit fast mode (less than 10 missing strides)
%% Adding constant and linear Inputs
Uf=[Uf;ones(size(Uf))]; %3-input system
%% Split data:
Yff{1}=Yf(:,1:900); %150 of Base + 750 of Adapt
Uff{1}=Uf(:,1:900);
Yff{2}=Yf(:,901:end); %150 of Adapt + 600 of Post
Uff{2}=Uf(:,901:end); %150 of Adapt + 600 of Post

%% Flat model:
[J,B,C,D,Q,R]=getFlatModel(Yff{1},Uff{1}); %Base+Adapt, minus last 100 of Adapt
model{1,1}=autodeal(J,B,C,D,Q,R);
model{1,1}.name='Static, CV1';
[J,B,C,D,Q,R]=getFlatModel(Yff{2},Uff{2});
model{1,2}=autodeal(J,B,C,D,Q,R);
model{1,2}.name='Static, CV2';

%load ./app/data/EMrealDim_CV_APalt1.mat
%%
for D1=3:4
%% Identify
    tic
    opts.robustFlag=false;
    opts.outlierReject=false;
    opts.fastFlag=true;
    opts.logFlag=true;
    opts.indB=1;
    opts.indD=1:2;
    for k=1:2
        [fAh,fBh,fCh,D,fQh,R,fXh,fPh,logL,outLog]=randomStartEM(Yff{k},Uff{k},D1,20,opts); %Slow/true EM
        model{D1+1}.runtime=toc;
        [J,B,C,X,~,Q,P] = canonize(fAh,fBh,fCh,fXh,fQh,fPh);
        model{D1+1,k}=autodeal(J,B,C,D,X,Q,R,P,logL,outLog);
        model{D1+1,k}.name=['EM (' num2str(D1) ', CV' num2str(k) ')']; %Robust mode does not do fast filtering
    end
end
%% add extra elements in B for viz
for k=1:2
    for j=4:5
        model{j,k}.B=[model{j,k}.B zeros(size(model{j,k}.B,1),1)];
    end
end
save ./app/data/EMrealDim_CV_APalt1.mat
%% COmpare
%%Train set:
for k=1:2
    vizDataFit(model(1:5,k),Yff{k},Uff{k})
    set(gcf,'Name',['CV' num2str(k) ', training data'])
end
%% Test set:
for k=1:2
    [f1,f2]=vizDataFit(model(1:5,3-k),Yff{k},Uff{k})
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
