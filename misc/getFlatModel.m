function [J,B,C,D,Q,R]=getFlatModel(Y,U)
  bad=any(isnan(Y),1);
  Y=Y(:,~bad);
  U=U(:,~bad);
J=0;
B=zeros(1,size(U,1));
Q=0;
C=ones(size(Y,1),1);
D=Y/U;
res=Y-D*U;
R=res*res'/size(Y,2);
end
