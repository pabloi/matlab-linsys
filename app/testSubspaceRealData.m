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
%% NaN removing, for subspace method:
Yf2=substituteNaNs(Yf')';
%% Flat model:
[J,B,C,D,Q,R]=getFlatModel(Yf,Uf);
model{1}=autodeal(J,B,C,D,Q,R);
model{1}.name='Flat';
%%
for D1=1:5
%% Identify
    tic
    [A,B,C,D,X,Q,R]=subspaceIDv2(Yf2,Uf,D1);
    model{D1+1}.runtime=toc;
    [J,B,C,X,~,Q] = canonizev2(A,B,C,X,Q);
    logL=dataLogLikelihood(Yf,Uf,J,B,C,D,Q,R);
    model{D1+1}=autodeal(J,B,C,D,X,Q,R,logL);
    model{D1+1}.name=['Subspace (' num2str(D1) ')']; %Robust mode does not do fast filtering
end
%%
save SubspaceRealDimCompare.mat
%% COmpare
vizModels(model(1:4))
%%
%vizDataFit(model([4:-1:1]),Y,U)
%vizDataFit(model([4:-1:1]),Y_p,U_p)
vizDataFit(model(1:4),Yf,Uf)
