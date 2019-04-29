function [Ap,pB]=projectObliq(A,B,C)
  
  %One-line execution: (not as efficient, pinv(B) is computed twice:
  %Ap=(projectPerp(A,B)*pinv(projectPerp(C,B)))*C;
  
  %Multi-line, efficient:
  pB=pinv(B);
  Ap=((A-A*pB*B)*pinv(C-C*pB*B))*C;
end
