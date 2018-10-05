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
    opts.robustFlag=true;
    opts.Niter=1500;
    opts.outlierReject=false;
    opts.fastFlag=true;
    for k=1:2
        [fAh,fBh,fCh,D,fQh,R,fXh,fPh]=randomStartEM(Yf(:,k:2:end),Uf(:,k:2:end),D1,10,opts); %Slow/true EM
        logL=dataLogLikelihood(Y,U,fAh,fBh,fCh,D,fQh,R,fXh(:,1),fPh(:,:,1));
        model{D1+1}.runtime=toc;
        [J,B,C,X,~,Q,P] = canonizev2(fAh,fBh,fCh,fXh,fQh,fPh);
        model{D1+1,k}=autodeal(J,B,C,D,X,Q,R,P,logL);
        model{D1+1,k}.name=['EM (' num2str(D1) ', CV' num2str(k) ')']; %Robust mode does not do fast filtering
    end
end
%%
save EMrealDimCompare1500CVrob.mat
%% COmpare
%%Train set:
for k=1:2
vizDataFit(model(2:5,k),Yf(:,k:2:end),Uf(:,k:2:end))
end
%% Test set:
for k=1:2
vizDataFit(model(2:5,3-k),Yf(:,k:2:end),Uf(:,k:2:end))
end
