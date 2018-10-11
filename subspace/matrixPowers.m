function An=matrixPowers(a,n)
  s=sqrt(numel(a));
  A=reshape(a,s,s);
  for i=1:n
    An((i-1)*s+[1:s],:)=A^i;
  end
end
