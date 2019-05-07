function [cInvA,invA,L,sqrtD]=pinvldl(A)
%Computes the pseudo-inverse of an positive semidefinite matrix A, using
%its Cholesky decomposition, as given by mycholcov(). Returns both the
%pseudo inverse and the transpose Chol decomp of pinv(A) and the Chol
%decomp of A itself.
%See also: ldl, pinvchol, mycholcov, chol, invchol

[L,D]=ldl(A);
L(isnan(L))=0; %This happens if A has non-diagonal Inf values
invL=L\eye(size(L));
sqrtD=sqrt(diag(D));
dth=1e2*eps; %Threshold to say that an eigenvalue is truly non-zero.
sqrtD(abs(sqrtD)<dth)=0; %Thresholding eigenvalues too small in magnitude to be considered non-zero
cInvA=invL'./sqrtD';
cInvA(isnan(cInvA))=0; %Defining Inf*0=0

if nargout>1
    invA=cInvA*cInvA';
    invA(isnan(invA))=0; %Defining Inf*0=0
end
