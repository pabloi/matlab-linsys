function [L,r]=mycholcov(A)
%A Cholesky-like decomposition that accepts semidefinite matrices, and
%always returns a triangular matrix, unlike Matlab's cholcov().
%For under-rank matrices, this returns a rectangular factor of size rxn,
%where r is the rank of the matrix, and n is the size of A. This allows for
%fast computation of pseudo-inverse with backlash, something that can't
%happen if the matrix has all-0 rows.
%See also: cholcov, testMyCholCov, pinvchol

r=size(A,1);
[L,p]=chol(A); %This can handle infinite diagonal elements
if p~=0 %Not positive definite, need to complete
    %Algorithm adapted from: https://arxiv.org/pdf/0804.4809.pdf made more efficient  
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
        %L=triu(L1); %Slower assignment, but taking tril() ensures triangularity
        L=triu(L1(1:r,:));
    else
        L=zeros(0,r);
        r=0;
    end    
end

%ALT:
%[L,D]=ldl(A);
%L=L*sqrt(D);
%r=[];

% %Functionally equivalent built-in code, but doesn't guarantee triangular form:
% r=size(A,1);
% L=cholcov(A);
% if numel(L)==0
%     L=zeros(0,r);
% end
% r=size(L,1);
% return