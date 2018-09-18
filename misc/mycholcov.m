function [L,r]=mycholcov(A)
%A Cholesky-like decomposition that accepts semidefinite matrices, and
%always returns a triangular matrix, unlike Matlab's cholcov()
r=size(A,1);
[L,p]=chol(A);
if p~=0 %Not positive definite, need to complete
    dA=diag(A);
    %% Algorithm from: https://arxiv.org/pdf/0804.4809.pdf
    %Just made more efficient
    
    tol= min(abs(dA))*1e-9; 
    tol=sqrt(tol);
    n=r;
     %L2=zeros(n); 
     %L2(1:p-1,1:p-1)=L;
    
    L1=zeros(n); 
    r=0; 
    for k=1:n %For each matrix row
        aux=A(k,:)-L1(:,k)'*L1;
        %aux=A-L1'*L1;
        a=sqrt(aux(k));
        if a>tol 
            r=r+1; 
            L1(r,:)=aux/a; %Faster assignment, although rounding errors may make this non triangular
        end 
    end 
    %L=L(:,1:r)'; 
    %L=L1;
    L=triu(L1); %Faster assignment, taking tril() ensures triangularity
    
%     for k=1:n %For each matrix row
%         aux=A(:,k)-L1*L1(k,:)';
%         a=sqrt(aux(k));
%         if a>tol 
%             aux=aux/a; 
%             r=r+1; 
%             L1(:,r)=aux; %Faster assignment, although rounding errors may make this non triangular
%         end 
%     end 
%     %L=L(:,1:r)'; 
%     L=tril(L1)'; %Faster assignment, taking tril() ensures triangularity
end