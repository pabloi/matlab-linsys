function [x,P]=KFpredict(A,Q,x,P,b)
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
aux=mycholcov(P); %To ensure psd
Aa=A*aux';
P=Aa*Aa'+Q;
%P=A*P*A'+Q; %This is much faster than the Chol decomposition, but may add instability
end
