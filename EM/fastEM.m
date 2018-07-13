function [A,B,C,D,Q,R,X]=fastEM(Y,U,D1)
%Y is D1 x N
%U is D2 x N
[D2,N]=size(Y);
%Initialize guesses of C,D
D=Y/U;
[pp,cc,aa]=pca(Y-D*U,'Centered','off');
C=cc(:,1:D1);
X=pp(:,1:D1)';

%Now, iterate estimations of A,B and C,D
for k=1:5
	[A,B,Q] = estimateAB(X, U);
	[~,X]=fwdSim(U,A,B,zeros(D2,D1),zeros(D2,size(U,1)),zeros(D1,1));
	[C,D,R] = estimateCD(Y, X(:,1:end-1), U);
	X=(C\(Y-D*U));
end
