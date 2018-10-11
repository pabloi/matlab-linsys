A=randn(3)
B=.1*randn(3);
B=B*B'

n=5;
d=size(A,1);
An=matrixPowers(A,n);
I=eye(d);
powerEstimates=An*(I-B);

[A1,B1]=fitMatrixPowers(powerEstimates);
norm(A-A1,'fro')
norm(B-B1,'fro')
