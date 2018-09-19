function newState=updateState(state,in,A,B,Q,q)
if nargin<5
  q=mycholcov(Q); %Allows for semidef. cov, not necessarily square
end
newState=A*state+B*in+q'*randn(size(q,1),1);
end
