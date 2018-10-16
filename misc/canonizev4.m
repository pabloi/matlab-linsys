function [J,B,C,X,V,Q,P] = canonizev4(A,B,C,X,Q,P)
%Canonize returns the canonical form of the linear system given by
%A,B,C,D,X; scaling states to converge to 1 at infinity (steady-state)
%under a step input in the first input component
warning('Deprecated: use canonize')

if nargin<6
    P=zeros(size(A));
end
if nargin<5 || isempty(Q)
    Q=zeros(size(A));
end
if nargin<4 || isempty(X)
    X=zeros(size(A,1),1);
end
[J,B,C,X,V,Q,P] = canonize(A,B,C,X,Q,P,'canonical');
end
