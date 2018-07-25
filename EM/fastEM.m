function [A,B,C,D,Q,R,X,P]=fastEM(Y,U,Xguess)
%A fast pseudo-EM implementation to do LTI-SSM identification
%INPUT:
%Y is D2 x N
%U is D3 x N
%Xguess - Either the number of states for the system (if scalar) or a guess
%at the initial states of the system (if D1 x N matrix)
[D2,N]=size(Y);
P=[];

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

logl=nan(11,1);

%Now, iterate estimations of A,B and C,D

%Alternating:
% for k=1:size(logl,1)-1
% 	[A,B,Q] = estimateAB(X, U);
% 	[~,X2]=fwdSim(U,A,B,zeros(D2,D1),zeros(D2,size(U,1)),zeros(D1,1));
% 	[C,D,~] = estimateCD(Y, X2(:,1:end-1), U);
% 	X=(C\(Y-D*U));
% end

%Meged:
for k=1:size(logl,1)-1
	[A,B,Q] = estimateAB(X, U);
    [C,D,~] = estimateCD(Y, X, U);
	[~,X2]=fwdSim(U,A,B,zeros(D2,D1),zeros(D2,size(U,1)),zeros(D1,1));
    X1=(C\(Y-D*U));
    X=.5*(X1+X2(:,1:end-1));
end

%Re-estimate Q,R:
aux=(Y-C*X-D*U);
R=aux*aux'/size(aux,2);
R=R+1e-5*eye(size(R));