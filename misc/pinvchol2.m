function [cInvA,cA,invA,L,D]=pinvchol2(A,safeFlag)
%Computes the pseudo-inverse of an positive semidefinite matrix A, using
%its Cholesky decomposition, as given by mycholcov(). Returns both the
%pseudo inverse and the transpose Chol decomp of pinv(A) and the Chol
%decomp of A itself.
%See also: mycholcov, chol, invchol

if true || nargin>1 %Safe method
    [cA,L,D]=mycholcov2(A,true);
    invL=L\eye(size(L));
    cInvA=sqrt(1./diag(D))'.*invL';
    cInvA(isnan(cInvA))=0; %Defining Inf*0=0
else %Fast method
    cA=mycholcov2(A);
    cInvA=cA\eye(size(A));
    L=[];D=[];
end

if nargout>2
    invA=cInvA*cInvA';
    invA(isnan(invA))=0; %Defining Inf*0=0
end
