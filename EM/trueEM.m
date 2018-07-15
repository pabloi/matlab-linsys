function [A,B,C,D,Q,R,X]=trueEM(Y,U,D1)

[D2,N]=size(Y);
%Initialize guesses of A,B,C,D,Q,R
D=Y/U;
[pp,cc,aa]=pca(Y-D*U,'Centered','off');
C=cc(:,1:D1);
X=pp(:,1:D1)';
[A,B,Q] = estimateAB(X, U);
[~,X]=fwdSim(U,A,B,zeros(D2,D1),zeros(D2,size(U,1)),zeros(D1,1));
[C,D,R] = estimateCD(Y, X(:,1:end-1), U);

%Initialize x0,P0
x0=[];
P0=[];

%Now, do E-M
for k=1:10
	%E-step: compute the expectation of latent variables given current parameter estimates
	[X,Ps,Xf,Pf,Xp,Pp,rejSamples]=statKalmanSmoother(Y,A,C,Q/1.5,R,x0,P0,B,D,U);
    %Note this is an approximation of true E-step in E-M algorithm. The
    %E-step requires to compute the expectation of the likelihood of the data under the
    %latent variables = E(L(Y,X|params)), to then maximize it
    %whereas here we are computing E(X|params) to then maximize L(Y,E(X)|params)

	%M-step: find parameters A,B,C,D,Q,R that maximize likelihood of data
	[A,B,Q] = estimateAB(X, U);
	[C,D,R] = estimateCD(Y, X, U);
end
