function [C] = linearize(funHandle,x0)
%Computes first-order Taylor approximation to the function funHandle around
%x0, such that funHandle(x)~funHandle(x0)+C*(x-x0)+ O(2);

d=1e-3;
f0=funHandle(x0);
C=zeros(length(f0),size(x0,1));
for j=1:size(x0,1)
    x=x0;
    x(j)=x(j)+d;
    C(:,j)=(funHandle(x)-f0)/d;
end

end

