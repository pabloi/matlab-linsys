function [J,B,C,X,V] = canonize(A,B,C,X)
%Canonize returns the canonical form of the linear system given by
%A,B,C,D,X

%% Find linear transformation to Jordan's canonical form 
[V,J] = jordan(A); %J=V\A*V; %V*X'=X -> V*X'_+1 = X_+1 = (A*X +B*u) = A*V*X' +B*u => J*X' + B'*u
% Deal with complex solutions:
a=imag(diag(J)); b=real(diag(J));
if any(abs(a./b)>1e-15) %If there are (truly) complex eigen-values, will transform to the real-jordan form
    [~,J] = cdf2rdf(V,J);
else %This is to avoid numerical errors from negligible imaginary parts
    J=real(J);
end

%% Estimate X: arbitrary scaling: (there has to be a more efficient way to do it)
%Doing X1=V\X doesn't work well
X1=ones(size(X)); %A different initial condition will be needed if one of the true states has 0 initial value, and J is not diagonal.
for i=2:size(X1,2)
    X1(:,i)=J*X1(:,i-1);
end


%% Re-estimate C,B
B=V\B;
C=(C*X)/X1; 
X=X1;


end

