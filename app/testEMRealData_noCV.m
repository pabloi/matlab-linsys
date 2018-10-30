%%
addpath(genpath('./'))
%%
clear all
%% Load real data:
[Y,Yf,Ycom,Uf]=groupDataToMatrixForm(false,true);
Yf=Yf-nanmedian(Yf(1:150,:,:)); %Subtracting Baseline
Yf=nanmedian(Yf,3)'; %Median across subjects, using nanmedian to exploit fast mode (less than 10 missing strides)
%% Flat model:
[J,B,C,D,Q,R]=getFlatModel(Yf,Uf);
model{1,1}=autodeal(J,B,C,D,Q,R);
model{1,1}.name='Flat';
%%
for D1=3
%% Identify
    tic
    opts.robustFlag=false;
    opts.outlierReject=false;
    opts.fastFlag=true; %Cannot do fast for NaN filled data
    opts.logFlag=true;
        [fAh,fBh,fCh,D,fQh,R,fXh,fPh,logL,outLog]=randomStartEM(Yf,Uf,D1,20,opts); %Slow/true EM
        model{D1+1}.runtime=toc;
        [J,B,C,X,~,Q,P] = canonize(fAh,fBh,fCh,fXh,fQh,fPh);
        model{D1+1}=autodeal(J,B,C,D,X,Q,R,P,logL,outLog);
        model{D1+1}.name=['EM (' num2str(D1) ')']; %Robust mode does not do fast filtering
end
%%
save ./app/data/EMrealDim_noCV_noSqrt.mat
%% COmpare
%%Train set:
vizDataFit(model(4),Yf,Uf)
set(gcf,'Name',['training data'])
%% See models
vizModels(model(4))
vizSingleModel(model{4},Yf,Uf)
