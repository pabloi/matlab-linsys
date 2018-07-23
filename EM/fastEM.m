function [A,B,C,D,Q,R,X,P]=fastEM(Y,U,Xguess)
%A fast pseudo-EM implementation to do LTI-SSM identification
%INPUT:
%Y is D2 x N
%U is D3 x N
%Xguess - Either the number of states for the system (if scalar) or a guess
%at the initial states of the system (if D1 x N matrix)
[D2,N]=size(Y);

%Initialize guesses of C,D
D=Y/U;
if numel(Xguess)==1
    D1=Xguess;
    [pp,~,~]=pca(Y-D*U,'Centered','off');
    Xguess=pp(:,1:D1)';
else
    D1=size(Xguess,1);
end
C=(Y-D*U)/Xguess;
X=Xguess;

logl=nan(21,1);
%Now, iterate estimations of A,B and C,D
for k=1:size(logl,1)-1
	[A,B,Q] = estimateAB(X, U);
	[~,X2]=fwdSim(U,A,B,zeros(D2,D1),zeros(D2,size(U,1)),zeros(D1,1));
	[C,D,~] = estimateCD(Y, X2(:,1:end-1), U);
	X=(C\(Y-D*U));
    %aux=(Y-C*X-D*U);
    %R2=aux*aux'/size(aux,2);
    %R=.95*R2+.05*trace(R2)*eye(size(R2))/size(R2,1); 
    %logl(k)=dataLogLikelihood(Y,U,A,B,C,D,Q,R);
end

%Estimate R:
aux=(Y-C*X-D*U);
R2=aux*aux'/size(aux,2);
R=.99*R2+.01*trace(R2)*eye(size(R2))/size(R2,1); %Regularizing solution slightly
% Do actual optimal estim. of states, instead of using the the fast estimate
[X,P,Xf,Pf,Xp,Pp,rejSamples]=statKalmanSmoother(Y,A,C,Q,R,[],[],B,D,U);

%Re-estimate Q,R:
maxRcond=1e4;
aux=(Y-C*X-D*U);
R=aux*aux'/size(aux,2);
R=(1-1/maxRcond)*R+(1/maxRcond)*trace(R)*eye(size(R))/size(R,1); 
aux=(X(:,2:end)-A*X(:,1:end-1)-B*U(:,1:size(X,2)-1));
Q=aux*aux'/size(aux,2);
Q=(1-1/maxRcond)*Q+(1/maxRcond)*trace(Q)*eye(size(Q))/size(Q,1); 

%logl(end)=dataLogLikelihood(Y,U,A,B,C,D,Q,R,X);
%figure
%subplot(2,1,1)
%plot(logl)
