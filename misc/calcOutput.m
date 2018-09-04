function out=calcOutput(state,in,C,D,R)
  r=cholcov(R); %Allows for semidef. cov
out=C*state+D*in+r'*randn(size(r,1),1);
end
