function out=calcOutput(state,in,C,D,R,r)
if nargin<6
  r=mycholcov(R); %Allows for semidef. cov
end
out=C*state+D*in+r'*randn(size(r,1),1);
end
