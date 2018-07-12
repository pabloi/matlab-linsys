function [J,Xh,V,K] = estimateDynv5(X, realPolesOnly, U, J0)
%estimateDyn for a given vector X, it estimates matrices J,B,V such that
%Xh(:,i+1)=J*Xh(:,i) + K*U; Xh(:,1)=1; and X~V*Xh where J is Jordan Canonical Form
%INPUTS:
%X: D-dimensional time-series [NxD matrix] to be approximated with linear dynamics.
%realPolesOnly: boolean flag indicating if only real poles are to be considered (exponentially decaying terms)
%U: input matrix/vector of length N. If empty we assume U=0;
%J0: can be a scalar which indicates the dimension of J (square) or can be an initial guess of J [has to be square matrix].
%OUTPUTS:
%
%
%Changes in v5: Completely changed approach. Computing A,B for
%x(k+1)=Ax(k)+Bu(k)+w(k), non canonical. Then canonizing.
%See also: sPCAv6
% Pablo A. Iturralde - Univ. of Pittsburgh - Last rev: Jul 12 2018

%Find A,B:
N=size(X,1);
D=size(X,2);
if isempty(U) || size(U,1)~=N
    U=zeros(N,0);
end
XU=[X U];
AB=X(2:N,:)'/XU(1:N-1,:)';
A=AB(1:D,1:D);
B=AB(:,D+1);

%Find MLE noise realization:
W=X(2:N,:)-X(1:N-1,:)*A'-U(1:N-1,:)*B';

%Canonize: (todo)
%[J,K,~,X,V] = canonize(A,B,C,X);
J=A;
K=B;
V=eye(D);
Xh=X;


