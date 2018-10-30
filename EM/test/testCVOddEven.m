
%% Get data:
squareFlag=false;
[Y,U,A,B,C,D,Q,R,x0,Yoff]=generateSyntheticData(squareFlag);
J=A;
B=[B zeros(size(B))];
D=[D, Yoff];
trueModel=autodeal(J,B,C,D,Q,R,x0,Y,U);
trueModel.name='True';
trueModel=autodeal(J,B,C,D,Q,R,x0,Y,U);
%%
Nfolds=2;
%%
U=[U;ones(size(U))];
%% Run cross-validation:
% Flat model:
for i=1:Nfolds
  Yaux=nan(size(Y));
  Yaux(:,[i:Nfolds:end])=Y(:,[i:Nfolds:end]);
  [J,B,C,D,Q,R]=getFlatModel(Yaux,U);
  model{1,i}=autodeal(J,B,C,D,Q,R);
  model{1,i}.name=['Flat, fold ' num2str(i)];
end
  %load ./EM/test/EMsynth_CV2.mat
% Higer-order models:
opts.robustFlag=false;
opts.outlierReject=false;
opts.fastFlag=0; %Cannot do fast for NaN filled data
opts.logFlag=true;
opts.indD=1:2;
opts.indB=1;
opts.Niter=1000;
for D1=1:5%6:7%1:3%:6
    for k=1:Nfolds
      tic
        Yaux=nan(size(Y));
        Yaux(:,[k:Nfolds:end])=Y(:,[k:Nfolds:end]);
        [fAh,fBh,fCh,D,fQh,R,fXh,fPh,logL,outLog]=randomStartEM(Yaux,U,D1,20,opts); %Slow/true EM
        model{D1+1,k}.runtime=toc;
        [J,B,C,X,~,Q,P] = canonize(fAh,fBh,fCh,fXh,fQh,fPh);
        model{D1+1,k}=autodeal(J,B,C,D,X,Q,R,P,logL,outLog);
        model{D1+1,k}.name=['EM (' num2str(D1) ', fold ' num2str(k) ')']; %Robust mode does not do fast filtering
    end
end
%%
  %save ./EM/test/EMsynth_CV2.mat

%% Visualize: (test data)
for k=1:Nfolds
  Yaux=Y;
  Yaux(:,[k:Nfolds:end])=nan;
  vizDataFit(model(2:8,k),Yaux,U)
  set(gcf,'Name',['Test CV odd/even, testing data, fold ' num2str(k)])
end
%% Visualize: (train data)
for k=1:Nfolds
  Yaux=nan(size(Y));
  Yaux(:,[k:Nfolds:end])=Y(:,[k:Nfolds:end]);
  vizDataFit(model(2:8,k),Yaux,U)
  set(gcf,'Name',['Test CV odd/even, training data, fold ' num2str(k)])
end
%%
vizModels([{trueModel},model(4,1:2)])
