function [A,B,C,Q,X,P]=transform(V,A,B,C,Q,X,P)
%All arguments except V are optional. V must be an invertible matrix.
%Equivalent to ss2ss, but allows for more parameters

Q=V*Q*V';
P=V*P*V';
A=V*A/V;
X=V*X;
C=C/V;
B=V*B;

end