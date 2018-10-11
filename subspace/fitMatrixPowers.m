function [A,B]=fitMatrixPowers(powerEstimates,noBiasFlag)
%Given a matrix powerEstimates of dimensions (n.d x d), estimates A,B such that
% powerEstimates(((n-1)*d)+[1:d],:) = A^n * (I-B) is solved in the least-squares sense

if nargin<2
  noBiasFlag=false;
end
d=size(powerEstimates,2);
n=size(powerEstimates,1)/d; %Should be an integer
A0=powerEstimates(1:d,:);
B0=zeros(size(A0));
I=eye(size(A0));
opts=optimset('MaxFunEvals',1e4,'TolFun',1e-12,'Display','off');
if noBiasFlag
  x=fminunc(@(x) norm(powerEstimates-matrixPowers(x(1:d^2),n),'fro'),[A0(:)],opts);
else
  x=fminunc(@(x) norm(powerEstimates-matrixPowers(x(1:d^2),n)*(I-reshape(x(d^2+1:end),d,d)),'fro'),[A0(:); B0(:)],opts);
  B=reshape(x(d^2+1:end),d,d);
end
A=reshape(x(1:d^2),d,d);

end
