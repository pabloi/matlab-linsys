function [A,B,Q] = estimateAB(X, U)
%Find A,B:
N=size(X,2);
D=size(X,1);
if isempty(U) || size(U,2)~=N || all(U(:)==0)
    U=zeros(0,N);
end
XU=[X; U];
%AB=X(:,2:N)/XU(:,1:N-1); %What we'd like to do. But this may be ill-cond
P=XU*XU'; P=P + 1e-9*eye(size(P));
AB=X(:,2:N)*XU(:,1:N-1)'/P;
A=AB(1:D,1:D);
B=AB(:,D+1);

%Find MLE noise realization:
W=X(:,2:N)-A*X(:,1:N-1)-B*U(:,1:N-1);
Q=(W*W')/size(W,2);

%Regularizing solution slightly: 
Q=Q+1e-5*eye(size(Q));


