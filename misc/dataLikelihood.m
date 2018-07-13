function l=dataLikelihood(Y,U,A,B,C,D,Q,R,X)
%Evaluates the likelihood of the data under a given model

if nargin<9 || isempty(X)
    %Ideally: we integrate over all possible latent variables X
    %In practice: we compute the MLE of X, and compute likelihood using it.
    %Much simpler, although not as accurate.
    outRejFlag=[];
    constFun=[];
    [X,~,~,~,~,~,rejSamples]=statKalmanSmoother(Y,A,C,Q,R,x0,P0,B,D,U,outRejFlag,constFun);
end

error('TO DO') %See likelihood expression from Albert and Shadmehr 2017
end