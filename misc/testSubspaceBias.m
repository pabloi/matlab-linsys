%%
A=[.95,0; 0 , .999];
B=[.05; .001]; %Both states asymptote at 1
Ny=3;
C=randn(Ny,2);
D=randn(Ny,1);
Q=zeros(2);
R=.01*eye(Ny);
x0=[0;0];
U=[zeros(1,500), ones(1,1000)];
d=2;


%% Sim:
[Y,X]=fwdSim(U,A,B,C,D,x0,Q,R);

%%
Xnoisy=X+.1*randn(size(X));
Xerr=Xnoisy-X;
A_=(Xnoisy(:,2:end-1)-B*U(:,1:end-1))/Xnoisy(:,1:end-2) %Estimate of A, with known B
Abiased=A-A*(Xerr*Xerr')/(Xnoisy*Xnoisy') + (Xerr(:,2:end)*Xerr(:,1:end-1)')/(Xnoisy*Xnoisy') %Theoretical result

%% On the actual subspace ID: (traditional)
[A1,B1,C1,D1,X1,Q1,R1]=subspaceID(Y,U,d);
err1(:,k)=sort(eig(A1))-sort(eig(A));

[A1,~,~,X1,~,~] = canonizev4(A1,B1,C1,X1,Q1); %Trying to get canonical states
Xerr=X1-X(:,1:end-1); %An estimate of the state estimation error.
%Can't do exactly because states are not uniquely determined. This likely overestimates the true error
A1
Abiased=A-A*(Xerr*Xerr')/(X1*X1') + (Xerr(:,2:end)*Xerr(:,1:end-1)')/(X1*X1') %Theoretical result
