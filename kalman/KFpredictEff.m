function [x,Pinv]=KFpredictEff(A,Qinv,x,Pinv,b)
%predict implements Kalman's predict step
if nargin<5 || isempty(b)
  b=0;
end
x=A*x+b;
%P=A*P*A'+Q;
QiA=Qinv*A;
Pinv=Qinv - QiA*(Pinv+A'*QiA)*QiA';
end
