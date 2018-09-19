%%
addpath(genpath('../EM/')) %Adding the matlab-sysID toolbox to path, just in case
addpath(genpath('../kalman/'))
addpath(genpath('../aux/'))
addpath(genpath('../sim/'))
%%
clear all
%% Create model:
D1=2;
D2=180;
N=1000;
A=randn(D1);
A=[.97,0;0,.995];
A=jordan(A); %Using A in its jordan canonical form so we can compare identified systems, WLOG
%B=3*randn(D1,1);
%B=B./sign(B); %Forcing all elements of B to be >0, WLOG
B=(eye(size(A))-A)*ones(size(A,1),1); %WLOG, arbitrary scaling
U=[zeros(300,1);ones(N,1);zeros(N/2,1)]'; %Step input and then removed
C=randn(D2,D1);
D=randn(D2,1);
Q=eye(D1)*.0001;
R=eye(D2)*.01;
[A,B,C,~,~,Q] = canonizev2(A,B,C,Q,Q);
%% Simulate
NN=size(U,2);
x0=zeros(D1,1);
[Y,X]=fwdSim(U,A,B,C,D,x0,Q,R);
[Y1,X1]=fwdSim(U,A,B,C,D,x0,[],[]); %Noiseless simulation, for comparison
[Xs,Ps]=statKalmanSmoother(Y,A,C,Q,R,[],[],B,D,U,false); %Kalman smoother estimation of states, given the true parameters (this is the best possible estimation of states)
logL=dataLogLikelihood(Y,U,A,B,C,D,Q,R,Xs(:,1),Ps(:,:,1))
%% Identify 1alt: trueEM starting from true solution
% fastFlag=0;
% tic
% [fAh,fBh,fCh,fDh,fQh,fRh,fXh,fPh]=EM(Y,U,Xs,[],fastFlag);
% flogLh=dataLogLikelihood(Y,U,fAh,fBh,fCh,fDh,fQh,fRh,fXh(:,1),fPh(:,:,1))
% toc
% [fAh,fBh,fCh,fXh,fV,fQh] = canonizev2(fAh,fBh,fCh,fXh,fQh);
%% Identify 1alt: trueEM starting from non-true solution
fastFlag=0;
robustFlag=true;
tic
[fAh,fBh,fCh,fDh,fQh,fRh,fXh,fPh]=EM(Y,U,D1,[],fastFlag,robustFlag);
flogLh=dataLogLikelihood(Y,U,fAh,fBh,fCh,fDh,fQh,fRh,fXh(:,1),fPh(:,:,1))
toc
[fAh,fBh,fCh,fXh,fV,fQh] = canonizev2(fAh,fBh,fCh,fXh,fQh);

%%
tic
[Ah1,Bh1,Ch1,Dh1,Qh1,Rh1,Xh1,Ph1]=randomStartEM(Y,U,2,5,'fast');
logLh1=dataLogLikelihood(Y,U,Ah1,Bh1,Ch1,Dh1,Qh1,Rh1,Xh1(:,1),Ph1(:,:,1))
toc
[Ah1,Bh1,Ch1,Xh1,~,Qh1] = canonizev2(Ah1,Bh1,Ch1,Xh1,Qh1);
%%
% LDS.A=Ah;
% LDS.B=Bh;
% LDS.C=Ch;
% LDS.D=Dh;
% LDS.Q=Qh;
% LDS.R=Rh;
% LDS.x0=Xh(:,1);
% LDS.V0=Ph(:,:,1);
% llh=LikelihoodLDS(LDS,Y,U);
%% Cheng & Sabes %Too slow
% addpath(genpath('../ext/lds-1.0/'))
% LDS.A=eye(size(A));
% LDS.B=ones(size(B));
% LDS.C=randn(D2,D1);
% LDS.D=randn(D2,1);
% LDS.Q=eye(size(Q));
% LDS.R=eye(size(R));
% LDS.x0=zeros(D1,1);
% LDS.V0=1e8 * eye(size(A)); %Same as my smoother uses
% [LDS,Lik,Xcs,Vcs] = IdentifyLDS(2,Y,U,U,LDS);
% csLogLh=dataLogLikelihood(Y,U,LDS.A,LDS.B,LDS.C,LDS.D,LDS.Q,LDS.R,LDS.x0,LDS.V0)
% toc

%% COmpare
%Using muti reps
Ah_=Ah1;Bh_=Bh1;Ch_=Ch1; Dh_=Dh1; Qh_=Qh1; Rh_=Rh1; Xh_=Xh1; logLh_=logLh1;
%Override fastEM: (too ugly too look at)
%fAh=zeros(size(Ah));fBh=zeros(size(Bh));fCh=zeros(size(Ch)); fDh=zeros(size(Dh)); fQh=zeros(size(Qh)); fRh=zeros(size(Rh)); fXh=zeros(size(Xh)); flogLh=zeros(size(logLh));

figure;
[pp,cc,aa]=pca(Y,'Centered','off');
M=min(3,size(Y,1));
for kk=1:M
subplot(2*M,2,(kk-1)*2+1) %Output along first PC of true data
hold on
plot(cc(:,kk)'*(C*X(:,1:end-1)+D*U),'LineWidth',1)
plot(cc(:,kk)'*(Ch_*Xh_+Dh_*U),'LineWidth',1)
plot(cc(:,kk)'*(fCh*fXh+fDh*U),'LineWidth',1)
plot(cc(:,kk)'*(Y),'k','LineWidth',1)
set(gca,'ColorOrderIndex',1)
aux=sqrt(sum((cc(:,kk)'*(Y-C*X(:,1:end-1)-D*U)).^2));
aux1=sqrt(sum((cc(:,kk)'*(Y-Ch_*Xh_-Dh_*U)).^2));
aux2=sqrt(sum((cc(:,kk)'*(Y-fCh*fXh-fDh*U)).^2));
bar([1900],mean([aux]),'EdgeColor','none','BarWidth',100)
bar([2000],mean([aux1]),'EdgeColor','none','BarWidth',100)
bar([2100],mean([aux2]),'EdgeColor','none','BarWidth',100)
if kk==1
    title('Output projection over main PCs: proj(Y-CX-DU) (RMSE)')
end
if kk<M
    set(gca,'XTick',[])
end
end

% subplot(3,2,1) %Output along first PC of true data
% [pp,cc,aa]=pca(Y,'Centered','off');
% hold on
% plot(cc(:,1)'*(Y),'LineWidth',1)
% plot(cc(:,1)'*(Ch_*Xh_+Dh_*U),'LineWidth',1)
% plot(cc(:,1)'*(fCh*fXh+fDh*U),'LineWidth',1)
% title('Output projection over main PCs')

[Y2,X2]=fwdSim(U,Ah_,Bh_,Ch_,Dh_,x0,[],[]);
[Y3,X3]=fwdSim(U,fAh,fBh,fCh,fDh,x0,[],[]);
M=2;
for i=1:M
subplot(2*M,2,4+2*i-1) %States
hold on
%Smooth versions
set(gca,'ColorOrderIndex',1)
p1=plot(X1(i,:),'LineWidth',2,'DisplayName','Actual');
p2=plot(X2(i,:),'LineWidth',2,'DisplayName','trueEM');
p3=plot(X3(i,:),'LineWidth',2,'DisplayName','trueEM-trueStart');
axis([0 2000 -.5 1.5])
%Fitted versions:
set(gca,'ColorOrderIndex',1)
plot(X(i,:),'LineWidth',1)
plot(Xh_(i,:),'LineWidth',1)
plot(fXh(i,:),'LineWidth',1)
%plot(Xcs(i,:),'LineWidth',1)
title('States')
legend([p1 p2 p3])
end

subplot(4,2,2) %1-ahead Output RMSE
hold on
res=(Y(:,2:end)-C*(A*X(:,1:end-2)+B*U(:,1:end-1))-D*U(:,2:end));
aux=sqrt(sum(res.^2));
plot(aux)
res=(Y(:,2:end)-Ch_*(Ah_*Xh_(:,1:end-1)+Bh_*U(:,1:end-1))-Dh_*U(:,2:end));
aux1=sqrt(sum(res.^2));
%aux1=sqrt(sum((Y-Ch_*Xh_-Dh_*U).^2));
plot(aux1,'LineWidth',2)
res=(Y(:,2:end)-fCh*(fAh*fXh(:,1:end-1)+fBh*U(:,1:end-1))-fDh*U(:,2:end));
aux2=sqrt(sum(res.^2));
%aux2=sqrt(sum((Y-fCh*fXh-fDh*U).^2));
plot(aux2,'LineWidth',2)
title('1-ahead output error (innov.) (RMSE)')
set(gca,'ColorOrderIndex',1)
bar([1900],mean([aux]),'EdgeColor','none','BarWidth',100)
bar([2000],mean([aux1]),'EdgeColor','none','BarWidth',100)
bar([2100],mean([aux2]),'EdgeColor','none','BarWidth',100)
text(100,.2, 'Need to add likelihood measure')
axis([0 2200 .8 2])
axis tight
grid on

subplot(4,2,4) %Prediction error (to self)
hold on
w=X(:,2:end-1)-A*X(:,1:end-2)-B*U(:,1:end-1);
%aux=sqrt(sum((w).^2));
aux=sqrt(diag(w'*Q*w));
plot(aux)
w=Xh_(:,2:end)-Ah_*Xh_(:,1:end-1)-Bh_*U(:,1:end-1);
aux1=sqrt(diag(w'*Qh_*w));
%aux1=sqrt(sum((Xh_(:,2:end)-Ah_*Xh_(:,1:end-1)-Bh_*U(:,1:end-1)).^2));
plot(aux1,'LineWidth',1)
w=fXh(:,2:end)-fAh*fXh(:,1:end-1)-fBh*U(:,1:end-1);
aux2=sqrt(diag(w'*fQh*w));
%aux2=sqrt(sum((fXh(:,2:end)-fAh*fXh(:,1:end-1)-fBh*U(:,1:end-1)).^2));
plot(aux2,'LineWidth',1)
title('State prediction error (to self) (Mahalanobis)')
set(gca,'ColorOrderIndex',1)
bar1=bar([1900],mean([aux]),'EdgeColor','none','BarWidth',100);
text(1800,.4e-3,['LogL=' num2str(logL)],'Color',bar1.FaceColor)
bar2=bar([2000],mean([aux1]),'EdgeColor','none','BarWidth',100);
text(1900,.3e-3,['LogL=' num2str(logLh_)],'Color',bar2.FaceColor)
bar3=bar([2100],mean([aux2]),'EdgeColor','none','BarWidth',100);
text(2000,.2e-3,['LogL=' num2str(flogLh)],'Color',bar3.FaceColor)
axis([0 2200 0 .5e-3])
grid on

subplot(4,2,6) %Smooth state error RMSE
hold on
aux=0;%sqrt(sum((X-X1).^2));
aux1=sqrt(sum((X1-X2).^2));
aux2=sqrt(sum((X1-X3).^2));
for i=1:2
    set(gca,'ColorOrderIndex',1)
    p1=plot(aux,'LineWidth',1,'DisplayName','actual');
    p2=plot(aux1,'LineWidth',1,'DisplayName','trueEM');
    p3=plot(aux2,'LineWidth',1,'DisplayName','fastEM');
    set(gca,'ColorOrderIndex',1)
    bar1=bar([1900],mean([aux]),'EdgeColor','none','BarWidth',100);
    bar2=bar([2000],mean([aux1]),'EdgeColor','none','BarWidth',100);
    bar3=bar([2100],mean([aux2]),'EdgeColor','none','BarWidth',100);
end
axis([0 2200 0 .08])
title('Smooth true state errors (RMSE)')

subplot(4,2,8) %True state error RMSE
hold on
aux=sqrt(sum((X-X1).^2));
aux1=sqrt(sum((X-X2).^2));
aux2=sqrt(sum((X-X3).^2));
for i=1:2
    set(gca,'ColorOrderIndex',1)
    p1=plot(aux,'LineWidth',1,'DisplayName','actual');
    p2=plot(aux1,'LineWidth',1,'DisplayName','trueEM');
    p3=plot(aux2,'LineWidth',1,'DisplayName','fastEM');
    set(gca,'ColorOrderIndex',1)
    bar1=bar([1900],mean([aux]),'EdgeColor','none','BarWidth',100);
    bar2=bar([2000],mean([aux1]),'EdgeColor','none','BarWidth',100);
    bar3=bar([2100],mean([aux2]),'EdgeColor','none','BarWidth',100);
end
axis([0 2200 0 .2])
title('State errors to true smooth state (RMSE)')
