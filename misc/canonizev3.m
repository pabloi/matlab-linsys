function [J,B,C,X,V,Q,P] = canonizev3(A,B,C,X,Q,P)
%Canonize returns the canonical form of the linear system given by
%A,B,C,D,X; scaling C to have unity norm along each column

if nargin<6
    P=zeros(size(A));
end
if nargin<5 || isempty(Q)
    Q=zeros(size(A));
end
if nargin<4 || isempty(X)
    X=zeros(size(A,1),1);
end

%% Find linear transformation to Jordan's canonical form
% A2=A-diag(diag(A));
% if any(abs(A2(:))>1e-15) %Don't bother finding Jordan form if matrix is already diagonal
%     [V,~] = jordan(A); %J=V\A*V; %V*X'=X -> V*X'_+1 = X_+1 = (A*X +B*u) = A*V*X' +B*u => J*X' + K*u
%     %Deal with badly scaled V matrix:
%     V2=V;
%     V2=V2./sqrt(abs(diag(V)));
%     V2=V2./sqrt(abs(diag(V)))';
%     V=V2; %V2 still returns the Canonical jordan form, but is better scaled
%     J=V\A*V;
% else
%     V=eye(size(A));
%     J=A;
% end
%

[V,J]=eig(A);
% Deal with complex solutions, if they happen:
a=imag(diag(J)); b=real(diag(J));
if any(abs(a./b)>1e-15) %If there are (truly) complex eigen-values, will transform to the real-jordan form
    [V,D]=eig(A);
    [V,~] = cdf2rdf(V,D);
else %Ignore imaginary parts
    V=real(V);
    J=real(J);
end

%% Scale so states converge to 1 on single input system with u=1
%(this cannot be done always, need to check)
[J,K]=transform(inv(V),A,B);
scale=sqrt(sum(C.^2,1)).*sign(max(K,[],2))';
scale(scale==0)=1; %Otherwise the transform is ill-defined
V2=diag(scale);
V=V2/V;

%% Sort states by decay rates: (these are only the decay rates if J is diagonal)
[~,idx]=sort(diag(J)); %This works if the matrix is diagonalizable
V=V(idx,:);

%% Transform with all the changes:
[J,B,C,Q,X,P]=transform(V,A,B,C,Q,X,P);

end
