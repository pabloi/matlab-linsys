%%
addpath(genpath('../'))
%% Create model:
D1=2;
D2=100;
N=1000;
A=randn(D1);
if max(abs(eig(A)))>1
A=.98*A/max(abs(eig(A)));
end
A=[.95 0; 0 .98];
%B=3*randn(D1,1);
%B=B./sign(B); %Forcing all elements of B to be >0, WLOG
B=(eye(size(A))-A)*ones(size(A,1),1); %WLOG, arbitrary scaling
U=[zeros(300,1);ones(N,1);zeros(N/2,1)]'; %Step input and then removed
C=randn(D2,D1);
D=randn(D2,1);
Q=eye(D1)*.0005;
R=eye(D2)*.01;

%% Simulate
NN=size(U,2);
x0=zeros(D1,1);
[Y,X]=fwdSim(U,A,B,C,D,x0,Q,R);
[Y1,X1]=fwdSim(U,A,B,C,D,x0,[],[]); %Noiseless simulation, for comparison

%% Do estimates:
[Ae,Be,Qe]=estimateAB(X(:,1:end-1),U);
[Ce,De,Re]=estimateCD(Y,X(:,1:end-1),U);
%% Log-likelihood:
logL=dataLogLikelihood(Y,U,A,B,C,D,Q,R,X) %Actual
logLe=dataLogLikelihood(Y,U,Ae,Be,Ce,De,Qe,Re,X)
%% Display results
%% All
A
Ae
Q
Qe
B
Be
sum(diag(R))
sum(diag(Re))
% %% Invariances
% %Eigenvalues of A:
% eig(A)
% eig(Ae)
% %Projection of Q^{-1} onto B:
% B'*pinv(Q)*B
% Be'*pinv(Qe)*Be
% % R is invariant itself, showing only its trace for compactness
% sum(diag(R))
% sum(diag(Re))
% % CQC^T is invariant, also showing trace for compactness
% sum(diag(C*Q*C'))
% sum(diag(Ce*Qe*Ce'))
% 
% %CB is invariant
% 
% 
% %D is invariant