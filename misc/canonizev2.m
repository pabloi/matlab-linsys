function [J,K,C,X,V,Q] = canonizev2(A,B,C,Xa,Q)
%Canonize returns the canonical form of the linear system given by
%A,B,C,D,X; scaling C to have unity norm along each column

% %% Find linear transformation to Jordan's canonical form 
% [V,J] = jordan(A); %J=V\A*V; %V*X'=X -> V*X'_+1 = X_+1 = (A*X +B*u) = A*V*X' +B*u => J*X' + K*u
% % Deal with complex solutions:
% a=imag(diag(J)); b=real(diag(J));
% if any(abs(a./b)>1e-15) %If there are (truly) complex eigen-values, will transform to the real-jordan form
%     [V,J] = cdf2rdf(V,J);
% else %This is to avoid numerical errors from negligible imaginary parts
%     J=real(J);
% end
% 
% %% Re-estimate C,K
% K=V\B;
% C=C*V;
% scale=sqrt(sum(C.^2,1));
% C=C./scale;
% X=scale'.*(V\Xa);
% K=scale'.*K;

if nargin<5
    Q=zeros(size(J));
end
%Lazy way:
sys=ss(A,B,C,zeros(size(C,1),size(B,2)),1);
[csys,V]=canon(sys);
J=csys.A;
K=csys.B;

%Scale B at will
%TODO: formally, we can only scale the states whose evolution depends
%solely on themselves, and any other states (ie. non-diagonalizable states
%that form part of a Jordan block along with some other states) need to be
%normalized with a different criteria.
%m=max(abs(K),[],2);
%s=sign(K(abs(K)==m));
%scale=(eye(size(J))-J)*ones(size(J,1),1).*s./m;
scale=(eye(size(J))-J)\K(:,1);
V2=diag(1./scale);
csys = ss2ss(csys,V2); 
J=csys.A;
K=csys.B;
C=csys.C;
V=V2*V;
X=V*Xa;
Q=V*Q*V';

%Sort states according to increasing time-constants
[~,idx]=sort(diag(J)); %This works if the matrix is diagonalizable
J=J(idx,idx);
K=K(idx);
C=C(:,idx);
X=X(idx,:);
V=V(idx,:);
Q=Q(idx,idx);

end

