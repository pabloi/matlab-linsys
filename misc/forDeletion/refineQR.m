function [Qopt,Ropt]=refineQR(predError,predUncertainty,C,A,R)
%Finds the analytically optimal (in the log-likelihood of data sense)
%values for Q,R given the one-step ahead prediction errors and the C matrix.
%Naturally, the new Q,R values will generate approximately the same kalman filtered (or
%smoothed) states, such that the prediction errors are preserved 
%(not so for the prediction uncertainties).
%Exact preservation of filtered states requires exact preservation of
%kalman gain, which cannot be guaranteed for the general case.

%First, compute error covariance:
S=predError*predError'/size(predError,2);

%Second compute steady-state Kalman gain:
V=median(predUncertainty,3); %Could take steady-state value instead
sV=chol(V);
Csv=C*sV';
CVC=Csv*Csv';
oldK=V*C'/(CVC+R);
oldCK=CVC/(CVC+R);
Asv=A*sV';
Aksv=A*chol(oldK*C*sV'*sV)';
oldQ=sV'*sV - Asv*Asv' +Aksv*Aksv';


%Optimal value of steady-state prediction uncertainty:
newV=oldK*S/C'; %newV*C' = K*S, the issue is this equation may not have exact solutions
%Ensure symmetry:
sV=chol(newV);
newV=sV'*sV; 
Csv=sV*C';
%newCVC=Csv'*Csv;
%newVC=newV*C; %=K*S
%Csv=C*sV';
%newCVC=Csv*Csv';

%Optimal value of Q:
Asv=A*sV';
%sS=chol(S);
Aksv=A*chol(oldK*C*newV)';
Qopt=sV'*sV - Asv*Asv' +Aksv*Aksv';

%Optimal value of R:
Ropt=S-Csv'*Csv; %=I_CK*S; %Equivalent expression, numerically unstable (not symm)


%Check:
% norm(newV-newV','fro')/norm(newV,'fro') %This should be 0 
% norm(Ropt-Ropt','fro')/norm(Ropt,'fro') %This should be 0 
% newCK=newCVC/S;
% norm(newCK-oldCK,'fro')/norm(oldCK,'fro') %This should be 0 
% norm(S-Ropt-newCVC,'fro')/norm(S,'fro') %This should be 0 

%A different, more basic, approach:
%Decompose R into a C-potent (Rp) and a C-null (Rn) matrices: R=Rn+Rp and
%C'*Rn*C=0, and the column-spaces of Rp and Rn are orthogonal but their
%union is the whole space.
%Do nothing for Q
%[~,Sn]=decomp2(S,C);
%[~,Rn]=decomp2(R,C);
%Ropt=R-Rn+Sn; %We only change the C-null part of R, in a way that is guaranteed to be PSD
%Qopt=oldQ;
%This is guaranteed to improve the model logL GIVEN the estimated states
%and uncertainty. However, if we re-estimate them using the new value of R,
%there is no guarantee that the logL will go up, or at least not go down.
end

function [P,Bn]=decomp(B,A)
%Decomposes the square, psd matrix B into, B =A*P*P'*A' +Bn, where A'*Bn*A=0
%Thus: A'*B*A=A'*A*P*P'*A'*A
D=pinv(A);
Tp=chol(D*B*D');
P=Tp';
S=A*P;
Bn=B-S*S'; %If A is full-rank, then Bn=0

%Sanity check:
tol=1e-10;
if abs(sum(sum(A'*Bn*A))) > tol
    error(['Null part is not null: ' num2str(abs(sum(sum(A'*Bn*A))))])
end
end

function [P,Bn]=decomp2(B,A)
%Decomposes the square, psd matrix B into, B =pinv(A)'*P*P'*pinv(A) +Bn, where A'*Bn*A=0
%Thus: A'*B*A=P*P'
D=pinv(A);
Tp=chol(A'*B*A);
P=Tp';
S=D'*P;
Bn=B-S*S'; %If A is full-rank, then Bn=0

%Sanity check:
tol=1e-10;
if abs(sum(sum(A'*Bn*A))) > tol
    error(['Null part is not null: ' num2str(abs(sum(sum(A'*Bn*A))))])
end
end