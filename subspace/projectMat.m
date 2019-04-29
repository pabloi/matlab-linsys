function Ap=projectMat(A,B)
  Ap=(A*pinv(B))*B;
end
