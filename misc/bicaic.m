function [BIC,AIC,BICalt] = bicaic(model,Y,logL)

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

%Alternative heuristic approach: count only parameters different from 0
%Presumption is that parameters that are exactly 0 were fixed and not free.
Na=sum(sum(model.J~=0));
Nb=sum(sum(model.B~=0))-M;%One column of B can be arbitrarily scaled to be all ones, so not counting
Nc=sum(sum(model.C~=0));
Nd=sum(sum(model.D~=0));
Nq=sum(sum(triu(model.Q)~=0));
Nr=sum(sum(triu(model.R)~=0));
k=Na+Nb+Nc+Nd+Nq+Nr; %Model free parameters
k_alt=k+M*N-Nq-Nr-Na-Nb; %Counting parameters as a matrix factorization problem

BIC=2*logL-log(N)*k;
BICalt=2*logL-log(N)*k_alt;
%BIC1=???  %Chen and Chen 2008
AIC=2*logL-2*k;


end
