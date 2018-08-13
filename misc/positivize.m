function P=positivize(M)
%Given a square matrix M, finds the 'closest' symmetric positive semi-definite matrix P, in
%the sense that norm(P-M,'fro') is minimized.
[V,D] = eig(M);
D(D<0)=0;
P=V*D/V;
P=.5*(P+P'); %This gets rid of numerical issues that prevent the solution from being symmetric