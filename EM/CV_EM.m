function [models] = CV_EM(Y,U,D1,Nfolds,method)
%Cross-validated sPCA: folds are defined by taking 1-out-of-Nfold
%datapoints for each fold (regularly interleaved)
for i=1:Nfolds
    Yred=Y(:,i:Nfolds:end);
    %[C,J,X,B,D,r] = sPCAv5(Yred,dynOrder,forcePCS,nullBD,outputUnderRank);
    [A,B,C,D,Q,R,X,P]=randomStartEM(Yred,U,D1,[],method);
    %Transform A, B to describe dynamics for every sample, instead of every Nfolds samples.
    [A,B,Q]=upsample(Nfolds,A,B,Q);
    A=A^(1/Nfolds);
    
    %Re-compute X to get a state estimate for every sample, instead of 1-every-Nfolds
    Y2=nan(size(Y));
    Y2(:,i:Nfolds:end)=Yred;
    [X,P,Xf,Pf,Xp,Pp,rejSamples]=statKalmanSmoother(Y2,A,C,Q,R,[],[],B,D,U,[],[]); %Full smoothing (not fast). NaN samples are ignored (predict step only)
    Xn=nan(size(X,1),ceil(size(Y,2)/Nfolds)*Nfolds);
    
    %% Because we want to compare across the different cross-fittings, we need to agree on a canonical form:
    [J,B,C,X,~,Q] = canonizev2(A,B,C,Xn,Q);
    %%
    model.J=J;
    model.B=B;
    model.C=C;
    model.D=D;
    model.Q=Q;
    model.R=R;
    model.X=X;
    model.trainingData=Y;
    model.triainingInput=U;
    
    %%
    models{i}=model;
end
end

