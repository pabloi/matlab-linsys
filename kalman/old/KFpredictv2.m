function [x,sP,P]=KFpredictv2(A,Q,x,P,b)
%predict implements Kalman's predict step
if nargin<5 || isempty(b)
  b=0;
end
x=A*x+b;
P=A*P*A'+Q;
sP=decomposition(P,'chol','upper');
end
