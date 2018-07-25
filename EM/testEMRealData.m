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

Y=[median(dataSym{1},3); median(dataSym{2},3)]';
U=[zeros(size(dataSym{1},1),1);ones(size(dataSym{2},1),1)]';
%% Identify 0: handcrafted
[model] = sPCAv8(Y(:,51:950)',2,[],[],[]);
J=model.J;
C=model.C;
X=[zeros(2,size(dataSym{1},1)) model.X];
B=model.B;
D=model.D;
Q=1e-3*eye(2);
aux=Y-C*X-D*U;
R=aux*aux'/size(aux,2);
R=R+1e-5*eye(size(R));
slogLh=dataLogLikelihood(Y,U,J,B,C,D,Q,R,X,[])
%% Identify 1: fast EM
tic
[fAh,fBh,fCh,fDh,fQh,fRh,fXh,fPh]=fastEM(Y,U,2);
flogLh=dataLogLikelihood(Y,U,fAh,fBh,fCh,fDh,fQh,fRh,fXh,fPh)
toc
[fJh,fKh,fCh,fXh,fV,fQh] = canonizev2(fAh,fBh,fCh,fXh,fQh);
flogLh=dataLogLikelihood(Y,U,fJh,fKh,fCh,fDh,fQh,fRh,fXh,fPh)
%% Identify 2: true EM
tic
[Ah,Bh,Ch,Dh,Qh,Rh,Xh,Ph]=trueEM(Y,U,2);
logLh=dataLogLikelihood(Y,U,Ah,Bh,Ch,Dh,Qh,Rh,Xh,Ph)
toc
[Jh,Kh,Ch,Xh,V,Qh] = canonizev2(Ah,Bh,Ch,Xh,Qh);
logLh=dataLogLikelihood(Y,U,Jh,Kh,Ch,Dh,Qh,Rh,Xh,Ph)
%% COmpare
x0=zeros(2,1);
figure;
subplot(3,2,1) %Output along first PC of true data
[pp,cc,aa]=pca(Y,'Centered','off');
hold on
plot(cc(:,1)'*(Y),'LineWidth',1)
plot(cc(:,1)'*(Ch*Xh+Dh*U),'LineWidth',1)
plot(cc(:,1)'*(fCh*fXh+fDh*U),'LineWidth',1)
title('Output projection over main PCs')

[Y2,X2]=fwdSim(U,J,K,Ch,Dh,x0,[],[]);
[Y3,X3]=fwdSim(U,fJ,fK,fCh,fDh,x0,[],[]);
for i=1:2
subplot(3,2,2*i) %States
hold on
%Smooth versions
set(gca,'ColorOrderIndex',1)
p2=plot(X2(i,:),'LineWidth',2,'DisplayName','trueEM');
p3=plot(X3(i,:),'LineWidth',2,'DisplayName','fastEM');
axis([0 2000 -.5 1.5])
%Fitted versions:
set(gca,'ColorOrderIndex',1)
plot(Xh(i,:),'LineWidth',1)
plot(fXh(i,:),'LineWidth',1)
title('States')
legend([p1 p2 p3])
end

subplot(3,2,3) %Output RMSE
hold on
aux1=sqrt(sum((Y-Ch*Xh-Dh*U).^2));
plot(aux1,'LineWidth',2)
aux2=sqrt(sum((Y-fCh*fXh-fDh*U).^2));
plot(aux2,'LineWidth',2)
title('Output error (RMSE)')
set(gca,'ColorOrderIndex',1)
bar([2000],mean([aux1]),'EdgeColor','none','BarWidth',100)
bar([2100],mean([aux2]),'EdgeColor','none','BarWidth',100)
text(100,.2, 'Need to add likelihood measure')
axis([0 2200 .8 2])
grid on
subplot(3,2,5) %Smooth output RMSE
hold on
%aux=sqrt(sum((Y-Y1).^2));
%plot(aux)
aux1=sqrt(sum((Y-Y2).^2));
plot(aux1,'LineWidth',1)
aux2=sqrt(sum((Y-Y3).^2));
plot(aux2,'LineWidth',1)
title('Smooth output error (RMSE)')
set(gca,'ColorOrderIndex',1)
%bar1=bar([1900],mean([aux]),'EdgeColor','none','BarWidth',100);
%text(1600,4,['LogL=' num2str(logL)],'Color',bar1.FaceColor)
bar2=bar([2000],mean([aux1]),'EdgeColor','none','BarWidth',100);
text(1700,3.5,['LogL=' num2str(logLh)],'Color',bar2.FaceColor)
bar3=bar([2100],mean([aux2]),'EdgeColor','none','BarWidth',100);
text(1800,3,['LogL=' num2str(flogLh)],'Color',bar3.FaceColor)
axis([0 2200 .8 5])
grid on