function [J,K,C,X,V] = canonizev2(A,B,C,Xa)
%Canonize returns the canonical form of the linear system given by
%A,B,C,D,X; scaling C to have unity norm along each column

%% Find linear transformation to Jordan's canonical form 
[V,J] = jordan(A); %J=V\A*V; %V*X'=X -> V*X'_+1 = X_+1 = (A*X +B*u) = A*V*X' +B*u => J*X' + K*u
% Deal with complex solutions:
a=imag(diag(J)); b=real(diag(J));
if any(abs(a./b)>1e-15) %If there are (truly) complex eigen-values, will transform to the real-jordan form
    [V,J] = cdf2rdf(V,J);
else %This is to avoid numerical errors from negligible imaginary parts
    J=real(J);
end

%% Re-estimate C,K
K=V\B;
C=C*V;
scale=sqrt(sum(C.^2,1));
C=C./scale;
X=scale'.*(V\Xa);
K=scale'.*K;

end

