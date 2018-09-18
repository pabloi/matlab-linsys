function [out,state]=fwdSim(in,A,B,C,D,x0,Q,R)
[M,N]=size(in);
out=nan(size(C,1),N);
state=nan(size(A,1),N+1);
if nargin<6 || isempty(x0)
state(:,1)=zeros(size(A,1),1);
else
state(:,1)=x0;
end
if nargin<7 || isempty(Q)
    Q=zeros(size(A));
end
if nargin<8 || isempty(R)
   R=zeros(size(C,1)); 
end
q=mycholcov(Q);
r=mycholcov(R);
for k=1:N
  out(:,k)=calcOutput(state(:,k),in(:,k),C,D,[],r);
  state(:,k+1)=updateState(state(:,k),in(:,k),A,B,[],q);
end
end
