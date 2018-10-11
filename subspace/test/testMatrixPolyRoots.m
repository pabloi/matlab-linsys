%%testMatrixPolyRoots

%%
A=[.95,0,0;0,1.9,0;0,0,.7];
for i=1:2:11 %For even polynomial order, there is no unique solution even in the reals, as there is a sign indeterminancy of eigenvalues, choosing the positive solution by default
An=A^i;
w=[1 zeros(1,i-1)];
A_=matrixPolyRoots(An,w);
i
norm(polyvalm([w 0],A_)-An,'fro')
norm(A_-A,'fro')
end
