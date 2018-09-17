function [L,r]=mycholcov(A)
%A Cholesky-like decomposition that accepts semidefinite matrices, and
%always returns a triangular matrix, unlike Matlab's cholcov()

[L,r]=chol(A);
if r~=0 %Not positive definite
    dA=diag(A);
    %% Algorithm from: https://arxiv.org/pdf/0804.4809.pdf
    %Just made more efficient
    tol= min(abs(dA))*1e-9; 
    tol=sqrt(tol);
    n=size(A,1);
    L=zeros(n); 
    %L=sparse([],[],n,n,(n+1)*n/2);
    r=0; 

    for k=1:n %For each matrix row
        aux=A(:,k)-L*L(k,:)';
        a=sqrt(aux(k));
        if a>tol 
            if k<n 
                aux=aux/a; 
            end 
            r=r+1; 
            L(:,r)=aux; %Faster assignment, although rounding errors may make this non triangular
        end 
    end 
    %L=L(:,1:r)'; 
    L=tril(L)'; %Faster assignment, taking tril() ensures triangularity
end