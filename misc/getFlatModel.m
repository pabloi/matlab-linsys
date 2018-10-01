function [J,B,C,D,Q,R]=getFlatModel(Y,U)
J=0;
B=0;
Q=0;
C=ones(size(Y,1),1);
D=Y/U;
res=Y-D*U;
R=res*res'/size(Y,2);
end
