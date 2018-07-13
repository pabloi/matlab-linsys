function [C,D,R] = estimateCD(Y, X, U)
%Find C,D:
N=size(X,2);
D1=size(X,1);
if isempty(U) || size(U,2)~=N
    U=zeros(0,N);
end
XU=[X; U];
CD=Y/XU;
C=CD(:,1:D1);
D=CD(:,D1+1:end);

%Find MLE noise realization:
Z=Y-C*X-D*U;
R=cov(Z'); %TODO: Should be uncentered 2nd moments
