function logL=dataLogLikelihood(Y,U,A,B,C,D,Q,R,X)
%Evaluates the likelihood of the data under a given model

if nargin<9 || isempty(X)
    %Ideally: we integrate over all possible latent variables X
    %In practice: we compute the MLE of X, and compute likelihood using it.
    %Much simpler, although not as accurate.
    outRejFlag=[];
    constFun=[];
    [X,~,~,~,~,~,rejSamples]=statKalmanSmoother(Y,A,C,Q,R,[],[],B,D,U,outRejFlag,constFun);
end

%Transition likelihoods:
w=X(:,2:end)-A*X(:,1:end-1)-B*U(:,size(X,2)-1);
lX=-.5*sum(w.*(Q\w)) -.5*log(det(Q));
%Output likelihoods
z=Y-C*X(:,1:size(Y,2))-D*U(:,size(Y,2));
lY=-.5*sum(z.*(R\z)) -.5*log(det(R));

logL=sum(lX)+sum(lY);

%error('TO DO') %See likelihood expression from Albert and Shadmehr 2017
end