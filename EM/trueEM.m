function [A,B,C,D,Q,R,X]=trueEM(Y,U,D1,x0,P0)
%TODO: this works, but is too slow (requires many iterations)
%1) study convergence. is it slow or oscillating?
%2) I do not trust the current kalman smoother. Sometimes it does not seem
%to work, although I am not sure.
%3) Check Cheng and Sabes 2006 equations. Is this function consistent with
%those?
%4) Can the speed be improved?


[D2,N]=size(Y);
%Initialize guesses of A,B,C,D,Q,R
D=Y/U;
[pp,cc,aa]=pca(Y-D*U,'Centered','off');
C=cc(:,1:D1);
X=pp(:,1:D1)';
[A,B,Q] = estimateAB(X, U);
[~,X]=fwdSim(U,A,B,zeros(D2,D1),zeros(D2,size(U,1)),zeros(D1,1));
[C,D,R] = estimateCD(Y, X(:,1:end-1), U);

if nargin<4 || isempty(x0)
    %Initialize x0,P0
    x0=[];
    P0=[];
end

%Now, do E-M
for k=1:150
	%E-step: compute the expectation of latent variables given current parameter estimates
	[X,Ps,Xf,Pf,Xp,Pp,rejSamples]=statKalmanSmoother(Y,A,C,Q,R,x0,P0,B,D,U);
    %Note this is an approximation of true E-step in E-M algorithm. The
    %E-step requires to compute the expectation of the likelihood of the data under the
    %latent variables = E(L(Y,X|params)), to then maximize it
    %whereas here we are computing E(X|params) to then maximize L(Y,E(X)|params)

	%M-step: find parameters A,B,C,D,Q,R that maximize likelihood of data
	[A,B,Q] = estimateAB(X, U);
	[C,D,R] = estimateCD(Y, X, U);
end
[X,Ps,Xf,Pf,Xp,Pp,rejSamples]=statKalmanSmoother(Y,A,C,Q,R,x0,P0,B,D,U);
