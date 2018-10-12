function [cC,C]=myPSDsum(A,B,vB)
  %Sums two PSD matrices by successive rank-1 updates to the cholesky decomp of the first matrix
  %This is expensive, but guarantees that the sum is a PSD matrix, something that may not happen if we simply do A+B
  %It requires the second matrix (B) to be hermitian
  %If a third input argument is given, vB, it is presumed that B=vB*vB', and vB'*vB must be diagonal matrix (columns of vB are orthogonal). For example [V,D]=diag(B), and vB=V*sqrt(D) achieves this.
  %If the first argument is upper triangular, it is presumed to be chol(A) and given to save some computation.
  %Compare to the alternative approach of doing:
  %cA=chol(A); cB=chol(B); C=cA'*cA + cB'*cB; cC=chol(C);
  %This function is faster, but less precise, especially for dimensions where x'*A*x and x'*B*x are well defined (non-zero)
  %My intuition is that the accuracy issue stems from using svd()
  %It *may* be that this function is more accurate for dimensions where x'*A*x and x'*B*x~0
  %See testPSDsum

  if all(all(A==triu(A))) %Presumed A is already the cholesky decomp
    cC=A;
  else
    cC=mycholcov(A);
  end
  if size(cC,1)<size(cC,2) %Positive semi-def matrix
      cC=[cC;zeros(size(cC,2)-size(cC,1),size(cC,2))]; %Padding with zeros
  end

  if nargin<3
    %INPUT check:
    if any(any((B-B')<0))
      error('Second matrix i not hermitian.')
    end
    [~,D,V]=svd(mycholcov(B)); %Equivalent to: [V,D]=eig(B); D=sqrt(D);
    if any(diag(D)<0)
      error('Second matrix is not PSD.') %This can only happen for a hermitian matrix if we have numerical errors
    end
    vB=V*D';
  else
    tol=1e-9;
    if any(any(abs(vB'*vB-diag(diag(vB'*vB)))>tol))
      error('Third argument must have orthogonal columns');
    end
  end

  %Do successive rank-1 updates:
  for j=1:size(vB,2)
    cC=cholupdate(cC,vB(:,j));
  end

  %Compute C if requested
  if nargout>1
    C=cC'*cC;
  end
end
