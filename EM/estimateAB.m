function [A,B,Q] = estimateAB(X, U)
%Find A,B:
N=size(X,2);
D=size(X,1);
if isempty(U) || size(U,2)~=N
    U=zeros(0,N);
end
XU=[X; U];
AB=X(:,2:N)/XU(:,1:N-1);
A=AB(1:D,1:D);
B=AB(:,D+1);

%Find MLE noise realization:
W=X(:,2:N)-A*X(:,1:N-1)-B*U(:,1:N-1);
Q=cov(W'); %TODO: It really should be the squared sums of W (uncentered)


