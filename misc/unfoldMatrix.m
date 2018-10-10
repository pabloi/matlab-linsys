function A=unfoldMatrix(An,w)
%Takes a matrix An that approximates the expansion w(1)*A^(n) + w(2)*A^(n-1)+..+w(n)*A
%And returns an estimate of A. Based on an eigenvalue decomposition
[v,d]=eig(An);
d=diag(d);
e=nan(size(d));
for l=1:length(d)
  %w(l)=fzero(@(x) polyval([ones(1,ord) -d(l)],x),1); %Can only be done for real functions
  rr=roots([w -d(l)]);
  aa=abs(imag(rr))./sqrt(imag(rr).^2 + real(rr).^2);
  [~,idx]=min(aa);
  e(l)=rr(idx); %Minimum phase solution (actually: solution closest to real line)
end
A=v*diag(e)/v; %The real() part needs to be here to avoid numerical issues in roots() not returning complex conjugate pairs of eigenvalues
end
