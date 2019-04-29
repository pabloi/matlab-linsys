function L=observabilityMatrix(A,C,N)
Nc=size(C,1);
for j=0:N-1
    L([1:Nc]+j*Nc,:)=C*A^j;
end

end

