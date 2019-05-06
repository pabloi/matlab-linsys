function [newX,newCholPt] = sqrtUpdatePredict(y,prevX, cholP, cholR, cholQ, A, C,b)
%Update+predict step (all in one) of the square-root formulation of Kalman filter using QR decomp
%INPUTS:
%prevX: predicted state from previous step
%cholP: cholesky-decomposition of predicted uncertainty from previous step (upper)
%cholR: cholesky decomp of R (upper triangular)
%cholQ: cholesty decomp of Q (upper)
%A:
%C:
%Following Park and Kailath 1995
% 
 Ny=size(cholR,1);
 Nx=size(cholQ,1);

M=[cholR C*cholP zeros(Ny,Nx); zeros(Nx,Ny) A*cholP cholQ];
[~,R]=qr(M',0);
cholSt=R(1:Ny,1:Ny);
newCholPt=R(Ny+1:end,Ny+1:end);
H=R(1:Ny,Ny+1:end);
newX=A*prevX+b+(cholSt\H)'*(y-C*prevX);
end

