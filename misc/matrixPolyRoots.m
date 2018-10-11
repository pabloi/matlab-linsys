function A=matrixPolyRoots(B,w,method)
%Solves the polynomial matrix equation B=w(1)*A^(n) + w(2)*A^(n-1)+..+w(n)*A
%Or, equivalently, finds the roots of w(1)*A^(n) + w(2)*A^(n-1)+..+w(n)*A-B=0;
%Based on an eigenvalue decomposition, returns one possible (non-unique) solution
%If possible, the matrix A returned will have real eigenvalues if B,w are real.
%Of all the possible solutions, returns the one with the smallest determinant(to be proven).

if nargin<3 || isempty(method)
  method='minDetReal';
end

[v,d]=eig(B);
d=diag(d);
e=nan(size(d));
for l=1:length(d)
  %w(l)=fzero(@(x) polyval([ones(1,ord) -d(l)],x),1); %Can only be done for real functions
  rr=roots([w -d(l)]);
  switch method
  case 'minPhase' %Choosing solution closest to real line
    aa=abs(imag(rr))./sqrt(imag(rr).^2 + real(rr).^2);
    [~,idx]=min(aa);
  case 'minDet'
    [~,idx]=min(abs(rr)); %Minimum abs(eigenvalue)
  case 'minDetReal'
    if any(imag(rr)==0) %There are real solutions, always the case for w, d(l) real and odd polynomial order
      rr=rr(imag(rr)==0); %Constraining to real solutions
    end
    [~,idx]=min(abs(rr)); %Minimum abs(eigenvalue)
  end
  e(l)=rr(idx);
  if any(abs(rr+e(l))<(100*d(l))*eps) %Some polynomials (such as B=A^2) allow for opposite solutions, choosing the first with non-neg real part
    warning('Found opposite roots for polynomial, this means there is no unique real-eigenvalue solution to the problem, returning the one with positive real parts')
      if real(e(l))<0
        e(l)=-e(l);
      end
  end
end
A=v*diag(e)/v; %The real() part needs to be here to avoid numerical issues in roots() not returning complex conjugate pairs of eigenvalues
end
