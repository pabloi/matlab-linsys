function [newX,newCholP] = sqrtUpdate(yt,prevXt, cholP, cholR, cholQ, At, Ct,bt)
%Update+predict step (all in one) of the square-root formulation of Kalman filter using QR decomp
%INPUTS:
%prevX: predicted state from previous step
%cholP: cholesky-decomposition of predicted uncertainty from previous step (upper)
%cholR: cholesky decomp of R (upper triangular)
%cholQ: cholesty decomp of Q (upper)
%A:
%C:
%Following Park and Kailath

Ny=size(cholR,1);
Nx=size(cholQ,1);
%preArray=[cholR	zeros(Ny,Nx);   cholP*Ct    cholP*At;   zeros(Nx,Ny)  cholQ];  
preArray=zeros(Ny+2*Nx,Ny+Nx);
preArray(1:Ny,1:Ny)=cholR;
preArray(Ny+1:Ny+Nx,:)=cholP*[Ct At];
preArray(Ny+Nx+1:end,Ny+1:Ny+Nx)=cholQ;
[~,postArray]=qr(preArray,0);
newCholP=postArray(Ny+1:Ny+Nx,Ny+1:Ny+Nx);
cholS=postArray(1:Ny,1:Ny); %Upper triangular
H=postArray(1:Ny,Ny+1:Ny+Nx);
%newX=A*prevX+b + H'*(cholS'\(y-C*prevX)); %This can be improved by using the Q matrix so no inversion is needed (cholS is triangular, so it does not matter much)
newX=prevXt*At+bt + ((yt-prevXt*Ct)/cholS)*H;
end

