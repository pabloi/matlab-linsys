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
%B=3*randn(D1,1);
%B=B./sign(B); %Forcing all elements of B to be >0, WLOG
B=(eye(size(A))-A)*ones(size(A,1),1); %WLOG, arbitrary scaling
U=[ones(N,1)]'; %Step input and then removed
C=randn(D2,D1);
D=randn(D2,1);
Q=eye(D1)*.0005;
R=eye(D2)*.01;

%% Simulate
NN=size(U,2);
x0=zeros(D1,1);
[Y,X]=fwdSim(U,A,B,C,D,x0,Q,R);
[Y1,X1]=fwdSim(U,A,B,C,D,x0,[],[]); %Noiseless simulation, for comparison
Xs=statKalmanSmoother(Y,A,C,Q,R,[],[],B,D,U,false); %Kalman smoother estimation of states, given the true parameters (this is the best possible estimation of states)
%% Identify 1: fast EM
tic
[fAh,fBh,fCh,fDh,fQh,fRh,fXh]=fastEM(Y,U,2);
toc
[fJ,fK,fCh,fXh,fV,fQh] = canonizev2(fAh,fBh,fCh,fXh,fQh);
%% Identify 2: true EM
tic
[Ah,Bh,Ch,Dh,Qh,Rh,Xh]=trueEM(Y,U,2,x0,zeros(size(A))); %Forcing initial state to all zeros, not necessary
toc
[J,K,Ch,Xh,V,Qh] = canonizev2(Ah,Bh,Ch,Xh,Qh);
%% Identify 3: sPCAv8"
[model] = sPCAv8(Y',2,false,false,false); %This assumes a single input with constant value
sAh=model.J;
sBh=model.B;
sCh=model.C;
sDh=model.D;
sQh=zeros(size(sAh)); %Smooth model
sXh=model.X;
[sJ,sK,sCh,sXh,sV,sQh] = canonizev2(sAh,sBh,sCh,sXh,sQh);

%% COmpare
figure;

[pp,cc,aa]=pca(Y,'Centered','off');

for i=1:3
    subplot(9,2,2*(i-1)+1) %Output along first PC of true data
    hold on
plot(cc(:,i)'*(Y),'LineWidth',1)
plot(cc(:,i)'*(Ch*Xh+Dh*U),'LineWidth',1)
plot(cc(:,i)'*(fCh*fXh+fDh*U),'LineWidth',1)
plot(cc(:,i)'*(sCh*sXh+sDh*U),'LineWidth',1)
if i==1
title('Output projection over main PCs')
end
end
[Y2,X2]=fwdSim(U,J,K,Ch,Dh,x0,[],[]);
[Y3,X3]=fwdSim(U,fJ,fK,fCh,fDh,x0,[],[]);
[Y4,X4]=fwdSim(U,sJ,sK,sCh,sDh,x0,[],[]);
for i=1:2
subplot(3,2,2*i) %States
hold on
%Smooth versions
set(gca,'ColorOrderIndex',1)
p1=plot(X1(i,:),'LineWidth',2,'DisplayName','Actual');
p2=plot(X2(i,:),'LineWidth',2,'DisplayName','trueEM');
p3=plot(X3(i,:),'LineWidth',2,'DisplayName','fastEM');
p4=plot(X4(i,:),'LineWidth',2,'DisplayName','sPCA');
axis([0 length(U) -.5 1.5])
%Fitted versions:
set(gca,'ColorOrderIndex',1)
plot(X(i,:),'LineWidth',1)
plot(Xh(i,:),'LineWidth',1)
plot(fXh(i,:),'LineWidth',1)
plot(sXh(i,:),'LineWidth',1)
title('States')
legend([p1 p2 p3 p4])
end

subplot(3,2,3) %Output RMSE
hold on
aux=sqrt(sum((Y-C*X(:,1:end-1)-D*U).^2));
plot(aux)
aux1=sqrt(sum((Y-Ch*Xh-Dh*U).^2));
plot(aux1,'LineWidth',2)
aux2=sqrt(sum((Y-fCh*fXh-fDh*U).^2));
plot(aux2,'LineWidth',2)
aux3=sqrt(sum((Y-sCh*sXh-sDh*U).^2));
plot(aux3,'LineWidth',2)
title('Output error (RMSE)')
set(gca,'ColorOrderIndex',1)
bar([length(U)+100],mean([aux]),'EdgeColor','none','BarWidth',100)
bar([length(U)+200],mean([aux1]),'EdgeColor','none','BarWidth',100)
bar([length(U)+300],mean([aux2]),'EdgeColor','none','BarWidth',100)
bar([length(U)+400],mean([aux3]),'EdgeColor','none','BarWidth',100)
text(100,.2, 'Need to add likelihood measure')
axis([0 length(U)+500 .8 2])
grid on
subplot(3,2,5) %Smooth output RMSE
hold on
aux=sqrt(sum((Y-Y1).^2));
plot(aux)
aux1=sqrt(sum((Y-Y2).^2));
plot(aux1,'LineWidth',1)
aux2=sqrt(sum((Y-Y3).^2));
plot(aux2,'LineWidth',1)
aux3=sqrt(sum((Y-Y4).^2));
plot(aux3,'LineWidth',1)
title('Smooth output error (RMSE)')
set(gca,'ColorOrderIndex',1)
bar([length(U)+100],mean([aux]),'EdgeColor','none','BarWidth',100)
bar([length(U)+200],mean([aux1]),'EdgeColor','none','BarWidth',100)
bar([length(U)+300],mean([aux2]),'EdgeColor','none','BarWidth',100)
bar([length(U)+400],mean([aux3]),'EdgeColor','none','BarWidth',100)
axis([0 length(U)+500 .8 5])
grid on

subplot(6,2,10) %Smooth state error RMSE
hold on
aux=X-X1;
aux1=(X-X2);
aux2=(X-X3);
aux3=(X-X4);
for i=1:2
    set(gca,'ColorOrderIndex',1)
    p1=plot(aux(i,:),'LineWidth',1,'DisplayName','actual');
p2=plot(aux1(i,:),'LineWidth',1,'DisplayName','trueEM');
p3=plot(aux2(i,:),'LineWidth',1,'DisplayName','fastEM');
p4=plot(aux3(i,:),'LineWidth',1,'DisplayName','sPCA');
    set(gca,'ColorOrderIndex',1)
    bar([length(U)+50*i],sqrt(mean(aux(i,:).^2)),'EdgeColor','none','BarWidth',50)
    bar([length(U)+100+50*i],sqrt(mean(aux1(i,:).^2)),'EdgeColor','none','BarWidth',50)
    bar([length(U)+200+50*i],sqrt(mean(aux2(i,:).^2)),'EdgeColor','none','BarWidth',50)
    bar([length(U)+300+50*i],sqrt(mean(aux3(i,:).^2)),'EdgeColor','none','BarWidth',50)
    axis([0 length(U)+500 -1 1])
end
title('Smooth state errors')
subplot(6,2,12) %Smooth state error RMSE
hold on
aux=X-X;
aux1=(X(:,1:end-1)-Xh);
aux2=(X(:,1:end-1)-fXh);
aux3=(X(:,1:end-1)-sXh);
for i=1:2
    set(gca,'ColorOrderIndex',1)
    p1=plot(aux(i,:),'LineWidth',1,'DisplayName','actual');
    p2=plot(aux1(i,:),'LineWidth',1,'DisplayName','trueEM');
    p3=plot(aux2(i,:),'LineWidth',1,'DisplayName','fastEM');
    p4=plot(aux3(i,:),'LineWidth',1,'DisplayName','sPCA');
    title('True state errors')
    set(gca,'ColorOrderIndex',1)
    bar([length(U)+50*i],sqrt(mean(aux(i,:).^2)),'EdgeColor','none','BarWidth',50)
    bar([length(U)+100+50*i],sqrt(mean(aux1(i,:).^2)),'EdgeColor','none','BarWidth',50)
    bar([length(U)+200+50*i],sqrt(mean(aux2(i,:).^2)),'EdgeColor','none','BarWidth',50)
    bar([length(U)+300+50*i],sqrt(mean(aux3(i,:).^2)),'EdgeColor','none','BarWidth',50)
    axis([0 length(U)+500 -1 1])
end