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
Y=median(Y,3)';
figure;
plot(sqrt(sum(diff(Y,[],2).^2,1))./sqrt(2*sum(Y(:,2:end).^2,1)),'DisplayName','Instant std, normalized');
hold on; plot(sqrt(sum(diff(Y,[],2).^2,1))/10,'DisplayName','Instant std')
legend
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
[J,B,C,D,Q,R]=getFlatModel(Yf,Uf);
model{1}=autodeal(J,B,C,D,Q,R);
model{1}.name='Flat';
%%
for D1=1:5
%% Identify
    tic
    opts.robustFlag=false;
    opts.outlierReject=false;
    opts.fastFlag=true;
    [fAh,fBh,fCh,D,fQh,R,fXh,fPh,logL]=randomStartEM(Yf,Uf,D1,20,opts); %Slow/true EM
    model{D1+1}.runtime=toc;
    [J,B,C,X,~,Q,P] = canonizev2(fAh,fBh,fCh,fXh,fQh,fPh);
    model{D1+1}=autodeal(J,B,C,D,X,Q,R,P,logL);
    model{D1+1}.name=['EM (iterated,all,' num2str(D1) ')']; %Robust mode does not do fast filtering
end
%%
save EMrealDimCompare1500v3.mat
%% COmpare
vizModels(model(1:5))
%%
%vizDataFit(model([4:-1:1]),Y,U)
%vizDataFit(model([4:-1:1]),Y_p,U_p)
vizDataFit(model(1:5),Yf,Uf)
