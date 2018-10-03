function [BIC,AIC] = bicaic(model,Y,U,logL)

M=size(model.J,1);
Ny=size(model.C,1);
Nu=size(model.D,2);
N=size(Y,2);

Na=M^2;
Nb=M*Nu;
Nc=Ny*M;
Nd=Ny*Nu;
Nq=M*(M+1)/2;
Nr=Ny*(Ny+1)/2;
k=Na+Nb+Nc+Nd+Nq+Nr; %Model free parameters

if nargin<4 %logL not given, computing
    method='approx';
    logL=N*Ny*dataLogLikelihood(Y,U,model.J,model.B,model.C,model.D,model.Q,model.R,[],[],method);
end
BIC=log(N)*k -2*logL;
%BIC1=???  %Chen and Chen 2008
AIC=2*k-2*logL;


end
