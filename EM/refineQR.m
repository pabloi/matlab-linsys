function [Qopt,Ropt]=refineQR(predError,predUncertainty,C,A,R)
%Finds the analytically optimal (in the log-likelihood of data sense)
%values for Q,R given the one-step ahead prediction errors and the C matrix.
%Naturally, the new Q,R values will generate the same kalman filtered (or
%smoothed) states, such that the prediction errors are preserved.

%First, compute error covariance:
S=predError*predError'/size(predError,2);

%Second compute steady-state Kalman gain:
V=mean(predUncertainty,3); %Could take steady-state value instead
CVC=C*V*C';
K=V*C'/(CVC+R);

%CK=CVC/(CVC+R);
I_CK=R/(CVC+R);
CiRC=C'*pinv(R)*C;
KC=(CiRC+pinv(V))\CiRC;
I_KC=(CiRC+pinv(V))\pinv(V);

%Optimal value of R:
%Ropt=(eye(size(R))-C*K)*S; %this should be symmetric
Ropt=I_CK*S; %this should be symmetric, but isn't (?)

%Optimal value of Q:
%Qopt=K*S*pinv(C') - A*K*C*pinv(C'*pinv(Ropt)*C)*A';
iC=pinv(C);
%Qopt=K*S*pinv(C)' - A*K*C*pinv(C)*Ropt*pinv(C)'*A';
Qopt=K*S*iC' - A*KC*iC*Ropt*iC'*A';
newV=K*S*iC';
Qopt=newV - A*I_KC*newV*A';
Qopt=newV - A*newV*A' -A*K*C*newV*A';
%Check:

newCVC=C*newV*C';
newK=newV*C'/(newCVC+Ropt);
norm(K-newK)/norm(K) %This should be 0 and is not
norm(S-Ropt-newCVC)/norm(S) %This should be 0 and is not

%Decompose R into a C-potent (Rp) and a C-null (Rn) matrices: R=Rn+Rp and
%C'*Rn*C=0, and the column-spaces of Rp and Rn are orthogonal but their
%union is the whole space.

% %First, find the optimal value of R:
% S=predError*predError'/size(predError,2);
% [~,Sn]=decomp2(S,C);
% %Ropt=S-C*mean(predUncertainty,3)*C'; %There is no guarantee this is PSD
% 
% %Second, find the C-null part of Ropt, R
% %iRopt=pinv(Ropt);
% %iR=pinv(R);
% %[Po,Rnopt]=decomp2(iRopt,C);
% %[P,Rn]=decomp2(iR,C);
% %Ropt=pinv(iR-Rn+Rnopt);
% 
% %[Po,Rnopt]=decomp2(Ropt,C);
% [P,Rn]=decomp2(R,C);
% Ropt=R-Rn+Sn; %We only change the C-null part of R, in a way that is guaranteed to be PSD
% 
% %Third, find optimal Q:
% %S=Po/P;
% %Qopt=S*Q*S';
% %(iQopt+C'*pinv(Ropt)*C) \(C'*pinv(Ropt)*C) = (iQ+C'*pinv(R)*C)\(C'*pinv(R)*C)
% %(C'*pinv(Ropt)*C) = (iQopt+C'*pinv(Ropt)*C)*((iQ+C'*pinv(R)*C)\(C'*pinv(R)*C))
% %iQopt+C'*pinv(Ropt)*C = ((C'*pinv(Ropt)*C)/ (C'*pinv(R)*C)) * (iQ+C'*pinv(R)*C)
% %iQopt= (C'*iRopt*C)*((C'*iR*C)\(iQ+C'*iR*C)-eye);
% %iQopt= (C'*iRopt*C)*((C'*iR*C)\pinv(Q));
% %iQopt= (Po*Po')/(P*P')*pinv(Q);
% %T=chol(Q);
% %iQopt= (Po*Po')/(P*P')*pinv(Q);
% %Qopt=pinv(iQopt);
% %PTP=Po\(P*T');
% %Qopt=PTP*PTP';
% Qopt=Q;
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