function [cA,L,D]=mycholcov2(A)
%A Cholesky-like decomposition that accepts semidefinite matrices.
%Returns a permuted unit triangular matrix.
%For under-rank matrices, cA will contain all-zero rows.
%This is different from mycholcov behavior, which guarantees triangularity
%but returns rectangular factors when A is underrank.
%See also: cholcov, mycholcov, testMyCholCov, pinvchol

%To do: efficiently check if A has Infinite non-diagonal elements. That would be bad (they can't be interpreted properly for inversion)
%Thus, any non-diagonal elements corresponding to a row or column with an Inf diagonal should be set to 0
if nargout<=1 %Just the cholesky decomposition
  [cA,p]=chol(A); %This can handle infinite diagonal elements, but NOT singular covariances. 3x slower than chol with single argument
end
dth=1e-10; %Threshold to say that an eigenvalue is truly non-zero.
if nargout>1 || p~=0 %Not positive definite, need to complete
  [L,D]=ldl(A); %This always works but is much slower than the algorithm for chol() so using only if necessary
  L(isnan(L))=0; %This happens if A has non-diagonal Inf values
  dd=diag(D);
  dd(abs(dd)<dth)=0; %Thresholding eigenvalues too small in magnitude to be considered non-zero
  %is a work-around to numerical issues when there is a truly 0 eigenvalue in the matrix. 
  if any(dd<0) %This can't happen if A is a covariance matrix
      error('mycholcov2:notPSD','Matrix was not PSD')
      %To do: potentially, I should check that D did not have 2 x 2 blocks,
      %something that happens if A had negative eigenvalues below the
      %tolerance (which hopefully never happens for covariance matrices)
  end
  cA=sqrt(dd).*L';
  cA(isnan(cA))=0; %Defining Inf*0=0
end
