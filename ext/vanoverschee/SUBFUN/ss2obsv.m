function [At,Bt,Ct,Dt,Kt]=ss2obsv(A,B,C,D,K)

[n,m]=size(B);
[l,n]=size(C);
if ~exist('K');K = [];end

% Observability matrix
cc=obsv(A,C);
T=inv(cc(1:n,:));
Ti=inv(T);
At=Ti*A*T;
Bt=Ti*B;
Ct=C*T;
Dt=D;
if K ~= [];Kt=Ti*K;else;Kt = zeros(n,l);end




