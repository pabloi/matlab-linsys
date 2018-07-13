function out=output(state,in,C,D,R)
out=C*state+D*in+sqrt(R)*randn(size(C,1),1);
end
