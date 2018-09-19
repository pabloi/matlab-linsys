function [cInvA,cA,invA]=invcov(A)
%Computes a modified inverse of the covariance matrix A. When the matrix is
%invertible, it returns the true inverse. Otherwise it returns the
%pseudoinverse substituting DIAGONAL zero elements with Inf.
%See also: mycholcov, pinvchol

[cInvA,cA,invA]=pinvchol(A); %Pseudo-inverse calculation
nullIdx=find(diag(A)==0); %Identify 0-covariance elements
invA(sub2ind(size(A),nullIdx,nullIdx))=Inf; %Substitute 0 elements with Inf