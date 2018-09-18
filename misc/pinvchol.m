function [cInvA,cA,invA]=pinvchol(A)
%Computes the transpose Chol decomp of the pseudo-inverse of an psd matrix A, 
%using its chol decomposition, as given by mycholcov(). 

[cA,r]=mycholcov(A);
%cA(abs(cA)<eps)=0; %Sometimes chol() manages to compute the Chol decomposition of practically null matrices
cInvA=mldivide(cA(1:r,:),eye(r));
%cInvA=pinv(cA(1:r,:));
% opts.UT = true;
% opts.TRANSA =false;
% n=size(U,1);
% cInvA=linsolve(U(1:r,:),eye(r),opts);
if nargout>2
    invA=cInvA*cInvA';
end