function [cA,L,D]=mycholcov2(A)
%A Cholesky-like decomposition that accepts semidefinite matrices, and
%always returns a triangular matrix, unlike Matlab's cholcov().
%For under-rank matrices, this returns a rectangular factor of size rxn,
%where r is the rank of the matrix, and n is the size of A. This allows for
%fast computation of pseudo-inverse with backlash, something that can't
%happen if the matrix has all-0 rows.
%See also: cholcov, testMyCholCov, pinvchol

%To do: efficiently check if A has Infinite non-diagonal elements. That would be bad (they can't be interpreted properly for inversion)
%Thus, any non-diagonal elements corresponding to a row or column with an Inf diagonal should be set to 0
%Further,
if nargout<=1 %Just the cholesky decomposition
  [cA,p]=chol(A); %This can handle infinite diagonal elements, but NOT singular covariances
end
if nargout>1 || p~=0 %Not positive definite, need to complete
  [L,D]=ldl(A);
  L(isnan(L))=0; %This happens if A has non-diagonal Inf values
  cA=sqrt(diag(D)).*L';
  cA(isnan(cA))=0; %Defining Inf*0=0
end
