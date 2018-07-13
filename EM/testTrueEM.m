%%
addpath(genpath('../')) %Adding the matlab-sysID toolbox to path, just in case
%% Create model:
D1=2;
D2=100;
N=1000;
A=randn(D1);
A=.98*A./max(abs(eig(A))); %Setting the max eigenvalue to .98
A=[.95,0;0,.99];
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
x0=zeros(D1,1);
[Y,X]=fwdSim(U,A,B,C,D,x0,Q,R);
%% Identify 1: fast EM
[fAh,fBh,fCh,fDh,fQh,fRh,fXh]=fastEM(Y,U,2);
[fJ,fK,fCh,fXh,fV] = canonizev2(fAh,fBh,fCh,fXh);
ss=sign(fK);
fK=fK.*ss;
fXh=fXh.*ss;
fK=ss'*fK*ss;
fCh=fCh.*ss';

%% Identify 2: true EM
[Ah,Bh,Ch,Dh,Qh,Rh,Xh]=trueEM(Y,U,2);
[J,K,Ch,Xh,V] = canonizev2(Ah,Bh,Ch,Xh);
ss=sign(K);
K=K.*ss;
Xh=Xh.*ss;
K=ss'*K*ss;
Ch=Ch.*ss';

%% COmpare
figure;
subplot(2,2,1) %Output along first PC of true data
[pp,cc,aa]=pca(Y,'Centered','off');
hold on
plot(cc(:,1)'*(Y),'LineWidth',1)
set(gca,'ColorOrderIndex',1)
plot(cc(:,1)'*(Ch*Xh+Dh*U),'LineWidth',2)
set(gca,'ColorOrderIndex',1)
plot(cc(:,1)'*(fCh*fXh+fDh*U),'-.','LineWidth',2)
title('Output projection over main PCs')

subplot(2,2,2) %States
hold on
plot(X','LineWidth',1)
set(gca,'ColorOrderIndex',1)
plot(Xh','LineWidth',2)
set(gca,'ColorOrderIndex',1)
plot(fXh','-.','LineWidth',2)
title('States')

subplot(2,2,3) %Output RMSE
hold on
set(gca,'ColorOrderIndex',1)
aux=sqrt(sum((Y-C*X(:,1:end-1)-D*U).^2));
plot(aux)
set(gca,'ColorOrderIndex',1)
aux1=sqrt(sum((Y-Ch*Xh-Dh*U).^2));
plot(aux1,'LineWidth',2)
set(gca,'ColorOrderIndex',1)
aux2=sqrt(sum((Y-fCh*fXh-fDh*U).^2));
plot(aux2,'-.','LineWidth',2)
title('Output error (RMSE)')
bar([1900:100:2100],mean([aux; aux1; aux2]'),'EdgeColor','none')
text(100,3, 'Need to add likelihood measure')

subplot(2,2,4) %States RMSE
title({'RMSE of states:'; 'need to define robust Canonical form'})