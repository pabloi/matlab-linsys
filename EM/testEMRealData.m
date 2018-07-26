%%
addpath(genpath('../')) %Adding the matlab-sysID toolbox to path, just in case
%% Load real data:
load ../data/dynamicsData.mat
addpath(genpath('./fun/'))
% Some pre-proc
B=nanmean(allDataEMG{1}(end-45:end-5,:,:)); %Baseline: last 40, exempting 5
clear data dataSym
for i=1:3 %B,A,P
    %Remove baseline
    data{i}=allDataEMG{i}-B;

    %Interpolate over NaNs
    for j=1:size(data{i},3) %each subj
    t=1:size(data{i},1); nanidx=any(isnan(data{i}(:,:,j)),2); %Any muscle missing
    data{i}(:,:,j)=interp1(t(~nanidx),data{i}(~nanidx,:,j),t,'linear','extrap'); %Substitute nans
    end
    
    %Compute asymmetry component
    aux=data{i}-fftshift(data{i},2);
    dataSym{i}=aux(:,1:size(aux,2)/2,:);
end

Y=[median(dataSym{1},3); median(dataSym{2},3);median(dataSym{3},3)]';
U=[zeros(size(dataSym{1},1),1);ones(size(dataSym{2},1),1);zeros(size(dataSym{3},1),1);]';
%%
Y=medfilt1([median(dataSym{1},3); median(dataSym{2},3)],5)';
U=[zeros(size(dataSym{1},1),1);ones(size(dataSym{2},1),1)]';
%% Identify 0: handcrafted
D1=3;
[model] = sPCAv8(Y(:,51:950)',D1,[],[],[]);
J=model.J;
C=model.C;
X=[zeros(size(model.X,1),size(dataSym{1},1)) model.X];
B=model.B;
D=model.D;
Q=1e-8*eye(size(X,1));
aux=Y-C*X-D*U;
R=aux*aux'/size(aux,2);
R=R+1e-5*eye(size(R));
[J,B,C,X,V,Q] = canonizev2(J,B,C,X,Q);
slogLh=dataLogLikelihood(Y,U,J,B,C,D,Q,R,X,[]);
%% Identify 1: fast EM
tic
%[fAh,fBh,fCh,fDh,fQh,fRh,fXh,fPh]=fastEM(Y,U,D1);
[fAh,fBh,fCh,fDh,fQh,fRh,fXh,fPh]=randomStartEM(Y,U,D1,10,'fast');
toc
[fJh,fKh,fCh,fXh,fV,fQh] = canonizev2(fAh,fBh,fCh,fXh,fQh);
flogLh=dataLogLikelihood(Y,U,fJh,fKh,fCh,fDh,fQh,fRh,fXh,fPh);
%% Identify 2: true EM
tic
[Ah,Bh,Ch,Dh,Qh,Rh,Xh,Ph]=trueEM(Y,U,D1);
toc
[Jh,Kh,Ch,Xh,V,Qh] = canonizev2(Ah,Bh,Ch,Xh,Qh);
logLh=dataLogLikelihood(Y,U,Jh,Kh,Ch,Dh,Qh,Rh,Xh,Ph);
%% COmpare
M=size(fXh,1);
x0=zeros(M,1);
figure;
[pp,cc,aa]=pca(Y,'Centered','off');

for kk=1:3
subplot(M+1,2,(kk-1)*2+1) %Output along first PC of true data
hold on
plot(cc(:,kk)'*(Ch*Xh+Dh*U),'LineWidth',1)
plot(cc(:,kk)'*(fCh*fXh+fDh*U),'LineWidth',1)
plot(cc(:,kk)'*(C*X+D*U),'LineWidth',1)
plot(cc(:,kk)'*(Y),'k','LineWidth',1)
end
title('Output projection over main PCs')

[Y2,X2]=fwdSim(U,Jh,Kh,Ch,Dh,x0,[],[]);
[Y3,X3]=fwdSim(U,fJh,fKh,fCh,fDh,x0,[],[]);

for i=1:M
subplot(M+1,2,2*i) %States
hold on
%Smooth versions
set(gca,'ColorOrderIndex',1)
p2=plot(X2(i,:),'LineWidth',2,'DisplayName','trueEM');
p3=plot(X3(i,:),'LineWidth',2,'DisplayName','fastEM');
p1=plot(X(i,:),'LineWidth',2,'DisplayName','sPCA');
axis([0 2000 -.5 1.5])
%Fitted versions:
set(gca,'ColorOrderIndex',1)
plot(Xh(i,:),'LineWidth',1)
plot(fXh(i,:),'LineWidth',1)
patch([1:size(Xh,2),size(Xh,2):-1:1]',[Xh(i,:)+squeeze(Ph(i,i,:))', fliplr(Xh(i,:)-squeeze(Ph(i,i,:))')]',p2.Color,'EdgeColor','none','FaceAlpha',.3)
title('States')
legend([p2 p3 p1])
end

subplot(M+1,2,2*M+1) %Smooth output RMSE
hold on
%aux=sqrt(sum((Y-Y1).^2));
%plot(aux)
aux1=sqrt(sum((Y-Y2).^2));
plot(aux1,'LineWidth',1)
aux2=sqrt(sum((Y-Y3).^2));
plot(aux2,'LineWidth',1)
aux0=sqrt(sum((Y-C*X-D*U).^2));
plot(aux0,'LineWidth',1)
title('Smooth output error (RMSE)')
set(gca,'ColorOrderIndex',1)
%bar1=bar([1900],mean([aux]),'EdgeColor','none','BarWidth',100);
%text(1600,4,['LogL=' num2str(logL)],'Color',bar1.FaceColor)
bar2=bar([2000],mean([aux1]),'EdgeColor','none','BarWidth',100);
text(1700,3.5,['LogL=' num2str(logLh)],'Color',bar2.FaceColor)
bar3=bar([2100],mean([aux2]),'EdgeColor','none','BarWidth',100);
text(1800,3,['LogL=' num2str(flogLh)],'Color',bar3.FaceColor)
bar0=bar([2200],mean([aux0]),'EdgeColor','none','BarWidth',100);
text(1900,2.5,['LogL=' num2str(slogLh)],'Color',bar0.FaceColor)
axis([0 2200 .0 4])
grid on