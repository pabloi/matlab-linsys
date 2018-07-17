function [A,B,C,D,Q,R,X]=fastEM(Y,U,D1)
%Y is D1 x N
%U is D2 x N
[D2,N]=size(Y);
%Initialize guesses of C,D
D=Y/U;
[pp,cc,aa]=pca(Y-D*U,'Centered','off');
C=cc(:,1:D1);
X=pp(:,1:D1)';

logl=nan(21,1);
%Now, iterate estimations of A,B and C,D
for k=1:size(logl,1)-1
	[A,B,Q] = estimateAB(X, U);
	[~,X2]=fwdSim(U,A,B,zeros(D2,D1),zeros(D2,size(U,1)),zeros(D1,1));
	[C,D,~] = estimateCD(Y, X2(:,1:end-1), U);
	X=(C\(Y-D*U));
    %aux=(Y-C*X2(:,1:end-1)-D*U);
    %R=aux*aux'/size(aux,2)-C*Q*C';
    %logl(k)=dataLogLikelihood(Y,U,A,B,C,D,Q,R);
end

%Estimate Q,R by reconciling the two approaches:
%aux=(X-X2(:,1:end-1));
%P=aux*aux'/size(aux,2);
%Q=P-A*P*A'; %Should roughly match with the estimate above
aux=(Y-C*X2(:,1:end-1)-D*U);
R1=aux*aux'/size(aux,2);
R=R1-C*Q*C';
aux=(Y-C*X-D*U);
R2=aux*aux'/size(aux,2);
R=.95*R2+.05*trace(R2)*eye(size(R2))/size(R2,1); %Regularizing solution slightly, so RCOND is never more than 20
% Do actual optimal estim. of states, instead of using the the fast estimate
[X,Ps,Xf,Pf,Xp,Pp,rejSamples]=statKalmanSmoother(Y,A,C,Q,R,[],[],B,D,U);
logl(end)=dataLogLikelihood(Y,U,A,B,C,D,Q,R,X);