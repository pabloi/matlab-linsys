function [J,B,C,X,V,Q,P] = canonizev5(N,A,B,C,X,Q,P)
%Canonize returns the canonical form of the linear system given by
%A,B,C,D,X; scaling states to reach the value of 1 at time N,
%under step input in the first input component
warning('Deprecated: use canonize')

if nargin<7
    P=zeros(size(A));
end
if nargin<6 || isempty(Q)
    Q=zeros(size(A));
end
if nargin<5 || isempty(X)
    X=zeros(size(A,1),1);
end
[J,B,C,X,V,Q,P] = canonize(A,B,C,X,Q,P,'canonical',N);
end
