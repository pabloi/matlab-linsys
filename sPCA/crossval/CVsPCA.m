function [models] = CVsPCA(Y,dynOrder,forcePCS,nullBD,outputUnderRank,Nfolds)
%Cross-validated sPCA: folds are defined by taking 1-out-of-Nfold
%datapoints for each fold (regularly interleaved)
for i=1:Nfolds
    %[C,J,X,B,D,r] = sPCAv5(Y(i:Nfolds:end,:),dynOrder,forcePCS,nullBD,outputUnderRank);
    [model] = sPCAv8(Y(i:Nfolds:end,:),dynOrder,forcePCS,nullBD,outputUnderRank);
    %Transform J, B to describe dynamics for every sample, instead of every Nfolds samples.
    model.J=model.J^(1/Nfolds);
    I=eye(size(model.J));
    model.B=(I-model.J^Nfolds)\((I-model.J)*model.B);
    %Re-define the initial state for folds that start at 2nd+ sample, such
    %that all folds impose the same initial condition for the first (true)
    %sample, even if they are not using it for fitting
%     newX0=model.X(:,1);
%     for k=1:i-1
%         newX0=model.J*newX0+model.B;
%     end
    newX0=model.J^(i-1) * model.X(:,1) + (I-model.J)\(I-model.J^(i-1))*model.B;
    [model.B,model.D,model.X]=chngInitState(model.J,model.B,model.C,model.D,model.X,newX0);
    
    %Re-compute X to get a state estimate for every sample, instead of 1-every-Nfolds
    Xn=nan(size(model.X,1),ceil(size(Y,1)/Nfolds)*Nfolds);
    for k=1:Nfolds
       if k==i
          Xn(:,[k:Nfolds:size(model.X,2)*Nfolds])=model.X;
       else
          Xn(:,[k:Nfolds:size(model.X,2)*Nfolds])=model.J^(k-i) * model.X + (I-model.J)\(I-model.J^(k-i))*model.B; 
       end
    end
    model.X=Xn(:,1:size(Y,1));
    
    %%
    models{i}=model;
end
end

