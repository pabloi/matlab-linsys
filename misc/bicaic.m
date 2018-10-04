function [BIC,AIC] = bicaic(model,logL)

M=size(model.J,1); %Number of states
Ny=size(model.C,1);
Nud=size(model.D,1);
Nub=size(model.B,1);
N=size(Y,2);

Na=M; %This presumes that we only count diagonal terms, as most models are diagonalizable.
%Slightly underestimates the number of params in a jordan non-diagonal form
Nb=M*(Nub-1); %One column of B can be arbitrarily scaled to be all ones, so not counting
Nc=Ny*M;
Nd=Ny*Nud;
Nq=M*(M+1)/2; %Symmetric matrix
Nr=Ny*(Ny+1)/2; %Symmetric matrix
k=Na+Nb+Nc+Nd+Nq+Nr; %Model free parameters

BIC=2*logL-log(N)*k;
%BIC1=???  %Chen and Chen 2008
AIC=2*logL-2*k;


end
