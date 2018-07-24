function logL=dataLogLikelihood(Y,U,A,B,C,D,Q,R,X,P)
%Evaluates the likelihood of the data under a given model

if nargin<9 || isempty(X) %|| isempty(P)
    %Ideally: we integrate over all possible latent variables X
    %In practice: we compute the MLE of X, and compute likelihood using it.
    %Much simpler, although not as accurate.
    outRejFlag=[];
    constFun=[];
    [X,~,~,~,~,~,rejSamples]=statKalmanSmoother(Y,A,C,Q,R,[],[],B,D,U,outRejFlag,constFun);
end

%Approximate logLikelihood:
w=X(:,2:end)-A*X(:,1:end-1)-B*U(:,1:size(X,2)-1);
logdetQ=sum(log(eig(Q))); %More efficient than .5*log(det(R)), no over/underflow issues. 
lX=-.5*sum(w.*(Q\w)) -logdetQ;
%Output likelihoods
z=Y-C*X(:,1:size(Y,2))-D*U(:,1:size(Y,2));
logdetR= sum(log(eig(R))); %More efficient than .5*log(det(R)), no over/underflow issues.
lY=-.5*sum(z.*(R\z)) -logdetR;
%Total:
logL=sum(lX)+sum(lY);

% %Alt: (exact?)
% logL=0;
% for i=1:size(Y,2)
%     z=Y(:,i)-C*X(:,i)-D*U(:,i);
%     T=C*P(:,:,i)*C'+R;
%     logL=logL+-.5*z'*(T\z) -.5*log(det(T));
% end

%error('TO DO') %See likelihood expression from Albert and Shadmehr 2017
end