function [T] = linearTransitionMatrix(N,a,q,b)
x=1:N;
T=@(u) exp((x'-a*x-b*u)/(2*sqrt(q)));

end

