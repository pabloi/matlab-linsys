function [cInvA,cA,invA]=pinvchol(A)
%Computes the pseudo-inverse of an positive semidefinite matrix A, using 
%its Cholesky decomposition, as given by mycholcov(). Returns both the 
%pseudo inverse and the transpose Chol decomp of pinv(A) and the Chol 
%decomp of A itself.
%See also: mycholcov, chol, invchol

[cA,r]=mycholcov(A);
%cA(abs(cA)<eps)=0; %Sometimes chol() manages to compute the Chol decomposition of practically null matrices
cInvA=mldivide(cA(1:r,:),eye(r));
%ALT:
% [L,D]=ldl(A);
% d=diag(D);
% sd=sqrt(1./d);
% aux=d<1e-9; %Threshold for PD (below this, the corresponding eigenvalue is considered to be 0)
% if any(aux)
%     sd(aux)=0; %pseudo-inverse
% end
% cInvA=L'\diag(sd);

if nargout>2
    invA=cInvA*cInvA';
end