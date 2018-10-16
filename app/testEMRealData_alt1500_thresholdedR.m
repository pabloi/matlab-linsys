%%
addpath(genpath('./'))
addpath(genpath('../robustCov/'))
%%
clear all
%% Load real data:
[Y,Yf,Ycom,Uf]=groupDataToMatrixForm();
Yf=Yf-nanmedian(Yf(1:50,:,:)); %Subtracting Baseline
Yf=median(Yf,3)'; %Median across subjects
Yf=Yf(:,1:1350); %Using only 400 of Post
Uf=Uf(:,1:1350);
%% Show data has signal-dependent noise
%Y=median(Y,3)';
%figure;
%plot(sqrt(sum(diff(Y,[],2).^2,1))./sqrt(2*sum(Y(:,2:end).^2,1)),'DisplayName','Instant std, normalized');
%hold on; plot(sqrt(sum(diff(Y,[],2).^2,1))/10,'DisplayName','Instant std')
%legend
%%
%%
load EMrealDimCompare1500v2.mat
model1=model; %Seeding
clear model
%% Flat model:
[J,B,C,D,Q,R]=getFlatModel(Yf,Uf);
model{1}=autodeal(J,B,C,D,Q,R);
model{1}.name='Flat';

%% Identify
for D1=1:4
    tic
    opts.robustFlag=false;
    opts.Niter=3500;
    opts.outlierReject=false;
    opts.fastFlag=true;
    opts.thR=.23; %Tuned by hand, by limiting the full optimal R by hand
    [fAh,fBh,fCh,D,fQh,R,fXh,fPh,logL]=randomStartEM(Yf,Uf,model1{D1+1}.X,10,opts); %Slow/true EM
    model{D1+1}.runtime=toc;
    [J,B,C,X,~,Q,P] = canonizev2(fAh,fBh,fCh,fXh,fQh,fPh);
    model{D1+1}=autodeal(J,B,C,D,X,Q,R,P,logL);
    model{D1+1}.name=['EM (iterated,all,' num2str(D1) ')']; %Robust mode does not do fast filtering
end
%%
save EMrealDimCompare1500_threshR.mat
%% COmpare
vizModels(model(4))
%%
%vizDataFit(model([4:-1:1]),Y,U)
%vizDataFit(model([4:-1:1]),Y_p,U_p)
vizDataFit(model(4),Yf,Uf)