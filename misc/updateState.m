function newState=updateState(state,in,A,B,Q)
  q=cholcov(Q); %Allows for semidef. cov, not necessarily square
newState=A*state+B*in+q'*randn(size(q,1),1);
end
