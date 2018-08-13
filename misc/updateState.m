function newState=updateState(state,in,A,B,Q)
newState=A*state+B*in+sqrt(Q)*randn(size(A,1),1);
end
