function H=myhankel(A,i,j)
  H=nan(i*size(A,1),j);
  for l=1:j
    a=A(:,l:l+i-1);
    H(:,l)=a(:);
  end
end
