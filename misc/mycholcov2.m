function [cA,L,D]=mycholcov2(A,safeFlag)
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
if true || nargin>1 %Safe method, works for PSD and numerically-nonPD matrices
    if nargout<=1 %Just the cholesky decomposition
      [cA,p]=chol(A); %This can handle infinite diagonal elements, but NOT singular covariances
    end
    if nargout>1 || p~=0 %Not positive definite, need to complete
      [L,D]=ldl(A); %This always works but is much slower than the algorithm for chol() so using only if necessary
      L(isnan(L))=0; %This happens if A has non-diagonal Inf values
      dd=diag(D);
      if any(dd<0) %This can't happen if A is a covariance matrix
          error('mycholcov2:notPSD','Matrix was not PSD')
          %To do: potentially, I should check that D did not have 2 x 2 blocks,
          %something that happens if A had negative eigenvalues below the
          %tolerance (which hopefully never happens for covariance matrices)
      end
      cA=sqrt(dd).*L';
      cA(isnan(cA))=0; %Defining Inf*0=0
    end
else
  cA=chol(A);
end
