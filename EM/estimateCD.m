function [C,D,R] = estimateCD(Y, X, U)
%Find C,D:
N=size(X,2);
D1=size(X,1);
if isempty(U) || size(U,2)~=N || all(U(:)==0)
    U=zeros(0,N);
end
XU=[X; U];
%CD=Y/XU; %What we'd like to do. But this may be ill-cond
P=XU*XU'; P=P + 1e-9*eye(size(P));
CD=Y*XU'/P;
C=CD(:,1:D1);
D=CD(:,D1+1:end);

%Find MLE noise realization:
Z=Y-C*X-D*U;
R=(Z*Z')/size(Z,2);

%Regularizing solution slightly:
maxRcond=1e4;
R=(1-1/maxRcond)*R+(1/maxRcond)*trace(R)*eye(size(R))/size(R,1); 