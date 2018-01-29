function [out,state]=fwdSim(in,A,B,C,D,x0)
[M,N]=size(in);
out=nan(size(C,1),N);
state=nan(size(A,1),N);
if nargin<5 || isempty(x0)
state(:,1)=zeros(size(A,1),1);
else
state(:,1)=x0;
end
for k=1:N
  out(:,k)=output(state(:,k),in(:,k),C,D);
  state(:,k+1)=updateState(state(:,k),in(:,k),A,B);
end
end
