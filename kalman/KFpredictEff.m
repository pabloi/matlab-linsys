function [x,iP,P]=KFpredictEff(A,iQ,x,iP,b)
%predict implements Kalman's predict step
if nargin<5 || isempty(b)
  b=0;
end
x=A*x+b;
iP=iQ-iP*A'*(iP+A'*iQ*A)*A*iP;
if nargout>2
    P=eye(size(iP))/iP;
end
end
