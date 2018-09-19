function [L,r]=mycholcov(A)
%A Cholesky-like decomposition that accepts semidefinite matrices, and
%always returns a triangular matrix, unlike Matlab's cholcov()
%See also: cholcov, testMyCholCov

r=size(A,1);
[L,p]=chol(A);
if p~=0 %Not positive definite, need to complete
    %Algorithm from: https://arxiv.org/pdf/0804.4809.pdf made more efficient  
    t=trace(A);
    if t>eps % Numeric precision
        n=r;
        tol= 1e-3*sqrt(t/n); 
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
        %L=L1(1:r,:); 
        %L=L1; %Equivalent to above, but faster.
        L=triu(L1); %Slower assignment, but taking tril() ensures triangularity
    else
        L=zeros(r);
        r=0;
    end    
end

% %Functionally equivalent built-in code, but doesn't guarantee triangular form:
% r=size(A,1);
% L=cholcov(A);
% if numel(L)==0
%     L=zeros(0,r);
% end
% r=size(L,1);
% return