function L=mycholcov(A)
%A Cholesky-like decomposition that accepts semidefinite matrices, and
%always returns a triangular matrix, unlike Matlab's cholcov()
%Copied from: https://arxiv.org/pdf/0804.4809.pdf
    % Full rank Cholesky factorization of A 
dA=diag(A); tol= min(dA(dA>0))*1e-9; 
n=size(A,1);
L=zeros(n); 
r=0; 
for k=1:n 
    r=r+1; 
    L(k:n,r)=A(k:n,k)-L(k:n,1:(r-1))*L(k,1:(r-1))'; 
    % Note: for r=1, the substracted vector is zero 
    if L(k,r)>tol 
        L(k,r)=sqrt(L(k,r)); 
        if k<n 
            L((k+1):n,r)=L((k+1):n,r)/L(k,r); 
        end 
    else 
        r=r-1; 
    end 
end 
L=L(:,1:r)'; 