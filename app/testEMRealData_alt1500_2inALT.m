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
Uf=[Uf;[zeros(size(Uf,1),1),abs(diff(Uf))]];
Uf(2,[351,651])=1;
%% Show data has signal-dependent noise
Y=median(Y,3)';
figure;
plot(sqrt(sum(diff(Y,[],2).^2,1))./sqrt(2*sum(Y(:,2:end).^2,1)),'DisplayName','Instant std, normalized');
hold on; plot(sqrt(sum(diff(Y,[],2).^2,1))/10,'DisplayName','Instant std')
legend
%% Flat model:
[J,B,C,D,Q,R]=getFlatModel(Yf,Uf);
model{1}=autodeal(J,B,C,D,Q,R);
model{1}.name='Flat';
%%
for D1=2:4
%% Identify
    tic
    opts.robustFlag=false;
    opts.Niter=1500;
    opts.outlierReject=false;
    opts.fastFlag=true;
    [fAh,fBh,fCh,D,fQh,R,fXh,fPh,logL]=randomStartEM(Yf,Uf,D1,10,opts); %Slow/true EM
    model{D1+1}.runtime=toc;
    [J,B,C,X,~,Q,P] = canonizev2(fAh,fBh,fCh,fXh,fQh,fPh);
    model{D1+1}=autodeal(J,B,C,D,X,Q,R,P,logL);
    model{D1+1}.name=['EM (iterated,all,' num2str(D1) ')']; %Robust mode does not do fast filtering
end
%%
save EMrealDimCompare1500_2inALT.mat
%% COmpare
vizModels(model(3:5))
%%
%vizDataFit(model([4:-1:1]),Y,U)
%vizDataFit(model([4:-1:1]),Y_p,U_p)
vizDataFit(model(3:5),Yf,Uf)
