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

%IMO: what we REALLY want to compute: p({y}|params) (as stated in Shadmehr and Albert 2017)
%What classical EM papers on LTI-SSM define as likelihood is: 
%p({y},{x}|params) = p(x0) \prod p(x_{k+1}|x_k y_k) \prod p(y_k|x_k) 
%See Cheng and Sabes 2006, Gharhamani and Hinton 1996, for example
%This is factorizable as: prior state prob, transition probs and observation probs
%However x is not known, so usually all these probabilitie are replaced by
%themselves CONDITIONAL on the observed data (Shumway and Stoffer 1982,
%although I am not sure how that works: conditional on the observed output,
%the observed output should have a huge likelihood!)

%The conditioning on observed data allows the replacement of {x} by the
%Kalman-smoother estimate 
%The approximation will be p({y},{x}|params) ~ p({y},\hat{x}|params), where \hat{x} is the kalman-smoother x estimate
%Does this work??? 

%Approximate logLikelihood: p({y},\hat{x}|params) -> Notice that this
%results in different likelihoods if we transform the state space and the
%parameters accordingly (i.e. not scale invariant). See Shumway and Stoffer
%--------------------------
%State transition likelihood
%w=X(:,2:end)-A*X(:,1:end-1)-B*U(:,1:size(X,2)-1);
%[D1,N1]=size(w);
%logdetQ=sum(log(eig(Q))); %More efficient than log(det(Q)), no over/underflow issues. 
%minus2lX=sum(w.*(Q\w)) +N1*logdetQ + N1*D1*log(2*pi); %This can't be, as is scale variant: if we rescale all the states to 1/10 (and all relate parameters accordingly, e.g. Q is 1/100 of prev value), thend
%Output likelihoods
%z=Y-C*X(:,1:size(Y,2))-D*U(:,1:size(Y,2));
%[D2,N2]=size(z);
%logdetR= sum(log(eig(R))); %More efficient than log(det(R)), no over/underflow issues.
%minus2lY=sum(z.*(R\z)) +N2*logdetR + N2*D2*log(2*pi);
%Total:
%logL=-.5*(sum(minus2lX)+sum(minus2lY));

%Approximate logLikelihood: p({y}|params)
z=Y-C*X(:,1:size(Y,2))-D*U(:,1:size(Y,2)); %Instead of X, this should be A*\hat{x} +B*u, where \hat{x} is the Kalman-filtered estimate [Albert and Shadmehr 2017, eq. A1.25]
[D2,N2]=size(z);
P=R+C*Q*C'; %Instead of Q, this should be A*P_{k-1}*A'+Q, where P_{k-1} is the uncertainty on state at time k-1 after the Kalman filtering (ie., only fwd pass) [Albert and Shadmehr 2017, eq. A1.25]
%This approximation should be exact in the limit Q-> 0 
logdetP= sum(log(eig(P))); 
minus2ly=sum(z.*(R\z)) +N2*logdetP + N2*D2*log(2*pi);
logL=-.5*(sum(minus2ly)); %This expression is scale invariant




% %Alt: (exact?)
% logL=0;
% for i=1:size(Y,2)
%     z=Y(:,i)-C*X(:,i)-D*U(:,i);
%     T=C*P(:,:,i)*C'+R;
%     logL=logL+-.5*z'*(T\z) -.5*log(det(T));
% end

%error('TO DO') %See likelihood expression from Albert and Shadmehr 2017
end