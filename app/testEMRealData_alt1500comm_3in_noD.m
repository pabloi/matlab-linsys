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
[Y,Ysym,Yf,Uf]=groupDataToMatrixForm();
Yf=Yf-nanmedian(Yf(1:50,:,:)); %Subtracting Baseline
Yf=median(Yf,3)'; %Median across subjects
Yf=Yf(:,1:1350); %Using only 400 of Post
Uf=Uf(:,1:1350);
Uf=[Uf;[zeros(size(Uf,1),1),abs(diff(Uf))]];
Uf=[Uf;[zeros(size(Uf(1,:),1),1),abs(diff(Uf(1,:)))]];
Uf(3,[351,651])=1;
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
for D1=1:4 %Just fourth order
%% Identify
    tic
    opts.robustFlag=false;
    opts.Niter=1500;
    opts.outlierReject=false;
    opts.fastFlag=true;
    opts.indD=1;
    for k=1:4 %Using 1,2,3 inputs
      if k<4
          [fAh,fBh,fCh,D,fQh,R,fXh,fPh,logL]=randomStartEM(Yf,Uf(1:k,:),D1,10,opts); %Slow/true EM
      else
          [fAh,fBh,fCh,D,fQh,R,fXh,fPh,logL]=randomStartEM(Yf,Uf([1,3],:),D1,10,opts); %Slow/true EM
      end
      model{D1+1,k}.runtime=toc;
      [J,B,C,X,~,Q,P] = canonizev2(fAh,fBh,fCh,fXh,fQh,fPh);
      model{D1+1,k}=autodeal(J,B,C,D,X,Q,R,P,logL);
      model{D1+1,k}.name=['EM (' num2str(D1) ')']; %Robust mode does not do fast filtering
    end
end
%%
save EMrealDimCompare1500comm_3in_noD.mat
