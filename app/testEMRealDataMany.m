%%
addpath(genpath('../aux/')) 
addpath(genpath('../kalman/')) 
addpath(genpath('../data/'))
addpath(genpath('../sPCA/'))
addpath(genpath('../../robustCov/')) 
%%
clear all
%% Load real data:
load ../data/dynamicsData.mat
addpath(genpath('./fun/'))
% Some pre-proc
B=nanmean(allDataEMG{1}(end-45:end-5,:,:)); %Baseline: last 40, exempting 5
clear data dataSym
subjIdx=2:16;
for i=1:3 %B,A,P
    %Remove baseline
    data{i}=allDataEMG{i}-B;

    %Interpolate over NaNs
    for j=1:size(data{i},3) %each subj
        t=1:size(data{i},1); nanidx=any(isnan(data{i}(:,:,j)),2); %Any muscle missing
        data{i}(:,:,j)=interp1(t(~nanidx),data{i}(~nanidx,:,j),t,'linear','extrap'); %Substitute nans
    end
    
    %Remove subj:
    data{i}=data{i}(:,:,subjIdx);
    
    %Compute asymmetry component
    aux=data{i}-fftshift(data{i},2);
    dataSym{i}=aux(:,1:size(aux,2)/2,:);
    
    
end

%% All data
Y=[median(dataSym{1},3); median(dataSym{2},3);median(dataSym{3},3)]';
U=[zeros(size(dataSym{1},1),1);ones(size(dataSym{2},1),1);zeros(size(dataSym{3},1),1);]';
%% Just B and A
Y=[median(dataSym{1},3); median(dataSym{2},3)]';
U=[zeros(size(dataSym{1},1),1);ones(size(dataSym{2},1),1)]';
%%
for i=1:size(dataSym{1},3)% all subjs
    Yall{i}=[dataSym{1}(:,:,i); dataSym{2}(:,:,i)]';
    Uall{i}=U;
end
%%
D1=2;
%% Identify 0: handcrafted sPCA
tic
model{1}= sPCAv8(Y(:,51:950)',D1,[],[],[]);
model{1}.runtime=toc;
model{1}.X=[zeros(size(model{1}.X,1),size(dataSym{1},1)) model{1}.X];
aux=Y-model{1}.C*model{1}.X-model{1}.D*U;
R=aux*aux'/size(aux,2);
model{1}.R=R+1e-8*eye(size(R));
model{1}.Q=zeros(D1);
logL=dataLogLikelihood(Y,U,model{1}.J,model{1}.B,model{1}.C,model{1}.D,model{1}.Q,model{1}.R,model{1}.X(:,1),model{1}.Q);
[J,B,C,X,~,Q] = canonizev2(model{1}.J,model{1}.B,model{1}.C,model{1}.X,model{1}.Q);
D=model{1}.D;
model{1}=autodeal(J,B,C,D,X,Q,R,logL);
model{1}.name='sPCA';
%% Identify 3: robust EM
for i=1:size(dataSym{1},3)% all subjs
    x0all{i}=model{1}.X;
end
tic
opts.robustFlag=true;
opts.Niter=1000;
[fAh,fBh,fCh,D,fQh,R,fXh,fPh,logL]=EM(Yall,Uall,x0all,opts); %Slow/true EM
model{2}.runtime=toc
[J,B,C,X,~,Q,P] = canonizev2(fAh,fBh,fCh,fXh,fQh,fPh);
model{2}=autodeal(J,B,C,D,X,Q,R,P,logL);
model{2}.name='EM (robust, smooth start)'; %Robust mode does not do fast filtering

%% COmpare
figure; 
subplot(2,1,2)
hold on;
for i=1:length(Yall)
   plot(sqrt((sum(Yall{i}-model{2}.C*model{2}.X{i}-model{2}.D*Uall{i}).^2))) 
end

subplot(2,1,2)
hold on;
for i=1:length(Yall)
   plot(sqrt((sum(Yall{i}-model{2}.C*model{2}.X{i}-model{2}.D*Uall{i}).^2))) 
end