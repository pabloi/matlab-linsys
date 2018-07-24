function [models] = CV_EM(Y,U,D1,Nfolds,method)
%Cross-validated sPCA: folds are defined by taking 1-out-of-Nfold
%datapoints for each fold (regularly interleaved)
for i=1:Nfolds
    %[C,J,X,B,D,r] = sPCAv5(Y(i:Nfolds:end,:),dynOrder,forcePCS,nullBD,outputUnderRank);
    [A,B,C,D,Q,R,X,P]=randomStartEM(Y,U,D1,[],method);
    %Transform A, B to describe dynamics for every sample, instead of every Nfolds samples.
    A=A^(1/Nfolds);
    I=eye(size(A));
    B=(I-A^Nfolds)\((I-A)*B); %Changing B such that the steady state response to a step input is unchanged.
    %newX0=A^(i-1) * X(:,1) + (I-A)\(I-A^(i-1))*B;
    %[B,D,X]=chngInitState(A,B,C,D,X,newX0);
    
    %Re-compute X to get a state estimate for every sample, instead of 1-every-Nfolds
    [X,P,Xf,Pf,Xp,Pp,rejSamples]=statKalmanSmoother(Y,A,C,Q,R,[],[],B,D,U);
    Xn=nan(size(X,1),ceil(size(Y,2)/Nfolds)*Nfolds);
    for k=1:Nfolds
       if k==i
          Xn(:,[k:Nfolds:size(X,2)*Nfolds])=X;
       else
          Xn(:,[k:Nfolds:size(Y,2)])=A^(k-i) * X(:,1:floor(size(Y,2)/Nfolds)) + B*U(:,[k:Nfolds:size(Y,2)]); 
       end
    end
    
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

