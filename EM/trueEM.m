function [A,B,C,D,Q,R,X,P]=trueEM(Y,U,Xguess,x0,P0)
%A true EM implementation to do LTI-SSM identification
%INPUT:
%Y is D2 x N
%U is D3 x N
%Xguess - Either the number of states for the system (if scalar) or a guess
%at the initial states of the system (if D1 x N matrix)

%TODO: this works, but is too slow (requires many iterations)
%1) Check Cheng and Sabes 2006 equations. Is this function consistent with
%those?
%2) Can the speed be improved?


[D2,N]=size(Y);
%Initialize guesses of A,B,C,D,Q,R
D=Y/U;
if numel(Xguess)==1
    D1=Xguess;
    [pp,~,~]=pca(Y-D*U,'Centered','off');
    Xguess=pp(:,1:D1)';
else
    D1=size(Xguess,1);
end
X=Xguess;

[A,B,Q] = estimateAB(X, U);
[~,X]=fwdSim(U,A,B,zeros(D2,D1),zeros(D2,size(U,1)),zeros(D1,1));
[C,D,R] = estimateCD(Y, X(:,1:end-1), U);

if nargin<4 || isempty(x0)
    %Initialize x0,P0
    x0=[];
    P0=[];
end

logl=nan(5,2);
%logl(1,1)=dataLogLikelihood(Y,U,A,B,C,D,Q,R,X);
%Now, do E-M
for k=1:size(logl,1)-1
	%E-step: compute the expectation of latent variables given current parameter estimates
	[X,P,Pt,~,~,~]=statKalmanSmoother(Y,A,C,Q,R,x0,P0,B,D,U);
    %l=dataLogLikelihood(Y,U,A,B,C,D,Q,R,X)
    %logl(k,2)=l;
    %Note this is an approximation of true E-step in E-M algorithm. The
    %E-step requires to compute the expectation of the likelihood of the data under the
    %latent variables = E(L(Y,X|params)), to then maximize it
    %whereas here we are computing E(X|params) to then maximize L(Y,E(X)|params)
    %logl(k,2)=dataLogLikelihood(Y,U,A,B,C,D,Q,R,X);
	%M-step: find parameters A,B,C,D,Q,R that maximize likelihood of data
    [A,B,C,D,Q,R,x0,P0]=estimateParams(Y,U,X,P,Pt);
    %l=dataLogLikelihood(Y,U,A,B,C,D,Q,R,X)
    %logl(k+1,1)=l;
    %[A,B,C,~,~,Q] = canonizev2(A,B,C,X,Q);
end
%figure
%subplot(2,1,1)
%plot(reshape(logl',numel(logl),1))
%subplot(2,1,2)
%plot(logl(:,2)-logl(:,1))
