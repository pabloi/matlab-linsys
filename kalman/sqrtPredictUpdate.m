function [newX,newCholPt,K,logL,rejectedFlag] = sqrtPredictUpdate(y,prevX, cPt, cRt, cQt, At, Ct,b,rejThresh)
%Predict+update step (all in one, in that order) of the square-root formulation of Kalman filter using QR decomp
%INPUTS:
%prevX: predicted state from previous step
%cholP: cholesky-decomposition of predicted uncertainty from previous step (upper)
%cholR: cholesky decomp of R (upper triangular)
%cholQ: cholesty decomp of Q (upper)
%A:
%C:
%Following Park and Kailath 1995
% 
 Ny=size(cRt,1);
 Nx=size(cQt,1);
rejectedFlag=false;
%M=[cholR C*A*cholP C*cholQ; zeros(Nx,Ny) A*cholP cholQ];
Mt=[cRt zeros(Ny,Nx); cPt*At*Ct cPt*At; cQt*Ct cQt];
%[~,R]=qr(M',0);
[~,R]=qr(Mt,0);
cholSt=R(1:Ny,1:Ny);
newCholPt=R(Ny+1:end,Ny+1:end);
H=R(1:Ny,Ny+1:end);
newX=At'*prevX+b;
K=(cholSt\H)';
innov=(y-Ct'*newX);
newX=newX+K*innov;
if rejThresh==0
    warning('sqrtPredictUpdate:lowRejectionThreshold','Rejection threshold was set to 0, all samples will be rejected')
end
if nargout>3 ||(nargin>8 && ~isempty(rejThresh))%Rejection computation or logL requested
    iLy=cholSt'\innov;
    z2=iLy'*iLy; %sum(iLy.^2,1); %z^2 scores
    if z2>rejThresh %Requested rejection and sample IS rejected, doing update only
       rejectedFlag=true;
       newX=At'*prevX+b;
       [~,newCholPt]=qr([cPt*At;cQt],0);
    else %Sample is accepted
        %nop
    end
    if nargout>3
        halfLog2Pi=0.91893853320467268;
        halfLogdetSigma= sum(log(diag(abs(cholSt))));
        logL=-.5*z2 -halfLogdetSigma-size(y,1)*halfLog2Pi;
    end
end

