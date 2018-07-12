%% Example script
addpath(genpath('./sPCA/'))
addpath(genpath('./sim/'))
%% Create model:
D1=2;
D2=100;
N=1000;
A=randn(D1);
A=.98*A./max(abs(eig(A))); %Setting the max eigenvalue to .98
A=[.99,0;0,.95];
A=jordan(A); %Using A in its jordan canonical form so we can compare identified systems, WLOG
B=3*randn(D1,1);
B=B./sign(B); %Forcing all elements of B to be >0, WLOG
U=[zeros(300,1);ones(N,1);zeros(N/2,1)]'; %Step input and then removed
C=randn(D2,D1);
C=C./sqrt(sum(C.^2,1));
D=randn(D2,1);
X0=randn(2,1);
Q=eye(D1)*.9;
R=eye(D2)*.4;

%% Simulate
NN=size(U,2);
X=nan(D1,NN);
Y=zeros(D2,NN);
X(:,1)=X0;
W=sqrt(Q)*randn(D1,NN); %Diagonal noise
Z=sqrt(R)*randn(D2,NN);
for i=2:(NN)
   X(:,i)=A*X(:,i-1)+B*U(:,i-1)+W(:,i-1);
   Y(:,i)=C*X(:,i)+D*U(:,i)+Z(:,i);
end

%% Identify 1: assume we know C,D
Xhat=C\(Y-D*U);
[J,Xh,V,K] = estimateDynv5(Xhat', [], U', []);

figure;
subplot(2,1,1)
hold on
plot(X')
plot(Xh,'o')
title('States')
subplot(2,1,2)
hold on
%Zh=
%Wh=
title('Noise realizations')

%% Identify 2: we do not know C,D (we do know D1);

%Initialize Ch, Dh, Xh:
Dh=Y/U;
[pp,cc,aa]=pca(Y-Dh*U,'Centered','off');
Ch=cc(:,1:D1);
Xh=pp(:,1:D1);

%Now, iterate estimations of A,B and C,D
for k=1:10
[Ah,Xh,~,Bh] = estimateDynv5(Xh, [], U', []);
[~,Xh]=fwdSim(U,Ah,Bh,zeros(D2,D1),zeros(D2,size(U,1)),zeros(D1,1));
XU=[Xh(:,1:NN);U];
CD=Y/XU;
Ch=CD(:,1:D1);
Dh=CD(:,D1+1:end);
Xh=(Ch\(Y-Dh*U))';

end
[J,K,Ch,Xh,V] = canonizev2(Ah,Bh,Ch,Xh');
ss=sign(K);
K=K.*ss;
Xh=Xh.*ss;
J=ss'*K*ss;


figure;
subplot(2,1,1)
hold on
plot(X'./sqrt(sum(X.^2,2))')
plot(Xh'./sqrt(sum(Xh.^2,2))','o')
title('States')
subplot(2,1,2)
hold on
%Zh=
%Wh=
title('Noise realizations')