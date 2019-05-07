function [cInvA,cA,invA,L,D]=pinvchol2(A)
%Computes the pseudo-inverse of an positive semidefinite matrix A, using
%its Cholesky decomposition, as given by mycholcov(). Returns both the
%pseudo inverse and the transpose Chol decomp of pinv(A) and the Chol
%decomp of A itself.
%Handles infinite/zero elements in A appropriately. Used in the backwards
%recursion of kalman filtering.
%See also: mycholcov, chol, invchol

[cA,L,D]=mycholcov2(A);
%opts.LT=true;
%invL=linsolve(L,eye(size(L)),opts);
invL=L\eye(size(L));
cInvA=sqrt(1./diag(D))'.*invL';
cInvA(isnan(cInvA))=0; %Defining Inf*0=0

if nargout>2
    invA=cInvA*cInvA';
    invA(isnan(invA))=0; %Defining Inf*0=0
end
