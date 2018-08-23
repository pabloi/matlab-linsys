function [Qopt,Ropt]=refineQR(predError,predUncertainty,C,Q,R)
%Finds the analytically optimal (in the log-likelihood of data sense)
%values for Q,R given the one-step ahead prediction errors and the C matrix.
%Naturally, the new Q,R values will generate the same kalman filtered (or
%smoothed) states, such that the prediction errors are preserved.

%Decompose R into a C-potent (Rp) and a C-null (Rn) matrices: R=Rn+Rp and
%C'*Rn*C=0, and the column-spaces of Rp and Rn are orthogonal but their
%union is the whole space.

D=pinv(C);
%First, find the optimal value of R:
S=predError*predError'/N2;
Ropt=S-C'*C*mean(predUncertainty,3)*C';

%Second, find the C-null part of Ropt, R
[Po,~]=decomp(Ropt,C);
[P,~]=decomp(R,C);

%Third, find optimal Q:
S=Po/P;
Qopt=S*Q*S';

end

function [P,Bn]=decomp(B,A)
%Decomposes B =A*P*P'*A' +Bn, where A'*Bn*A=0
D=pinv(A);
T=chol(B);
Tp=(D*T);
P=Tp';
S=A*P;
Bn=B-S*S'; %If A is full-rank, then Bn=0

%Sanity check:
tol=1e-10;
if abs(sum(sum(A'*Bn*A))) > tol
    error(['Null part is not null: ' num2str(abs(sum(sum(A'*Bn*A))))])
end
end