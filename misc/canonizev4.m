function [J,B,C,X,V,Q,P] = canonizev4(A,B,C,X,Q,P)
%Canonize returns the canonical form of the linear system given by
%A,B,C,D,X; scaling states to converge to 1 at infinity (steady-state)
%under a step input in the first input component

if nargin<6
    P=zeros(size(A));
end
if nargin<5 || isempty(Q)
    Q=zeros(size(A));
end
if nargin<4 || isempty(X)
    X=zeros(size(A,1),1);
end

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
scale=(eye(size(J))-J)\K(:,1);
V2=diag(1./scale);
V=V2/V;

%% Sort states by decay rates: (these are only the decay rates if J is diagonal)
[~,idx]=sort(diag(J)); %This works if the matrix is diagonalizable
V=V(idx,:);

%% Transform with all the changes:
[J,B,C,Q,X,P]=transform(V,A,B,C,Q,X,P);

end
