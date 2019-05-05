function [x,newcPt]=KFpredictSQRT(A,cQt,x,cPt,b)
%predict implements Kalman's predict step
%INPUTS: A,Q,b constitute the matrices from the update equation at this step: 
%x_{k+1} = A*x_k + b
%x is the current MLE estimate of x_k
%P is the uncertainty (Cov matrix) of the estimate
%%cholP is the cholesky decomposition of the uncertainty P of the estimate
if nargin<5 || isempty(b)
  b=0;
end
x=A*x+b;
[~,newcPt]=qr([cPt*A';cQt],0);
end
