function [Q,R]=learnQR(Y,A,C,b,d)
%Simple heuristics to estimate stationary matrices Q, R
%The idea is to smooth the output Y, and use the
%smoothed output as a proxy for C*X+d
%A better idea may be to do EM: estimate QR,
%then run the filter, re-estimate QR and so forth

if nargin<4 || isempty(b)
  b=0;
end
if nargin<5 || isempty(d)
  d=0;
end

[D,N]=size(Y);
M=5; %Smoothing window. The optimal value for this depends on A, naturally.
smY=conv2(Y,ones(1,M)/M,'same');
idx=((M-1)/2+1):(D-(M-1)/2);

w=Y(:,idx)-smY;
R=cov(w);
v=X(:,2:end)-A*X(:,1:end-1)-b;
Q=cov(v);
end
