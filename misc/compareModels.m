function [fh] = compareModels(model,Y,U)

M=size(model{1}.X,1);
fh=figure;
Ny=2;
Nx=M+4;

%Compute output and residuals
for i=1:length(model)
    model{i}.out=model{i}.C*model{i}.X+model{i}.D*U;
    model{i}.res=Y-model{i}.out;
    model{i}.oneAheadStates=model{i}.J*model{i}.X(:,1:end-1)+model{i}.B*U(:,1:end-1);
    model{i}.oneAheadOut=model{i}.C*(model{i}.oneAheadStates)+model{i}.D*U(:,2:end);
    [Y2,X2]=fwdSim(U,model{i}.J,model{i}.B,model{i}.C,model{i}.D,model{i}.X(:,1),[],[]);
    model{i}.smoothStates=X2;
    model{i}.smoothOut=Y2;
end

% Plot output PCs and residuals
[cc,pp,aa]=pca(Y','Centered','off');
for kk=1:M+2
    subplot(Nx,Ny,(kk-1)*Ny+1) %Output along first PC of true data
    hold on
     plot(cc(:,kk)'*Y,'k','LineWidth',1)
     set(gca,'ColorOrderIndex',1)
    for i=1:length(model)
        p(i)=plot(cc(:,kk)'*(model{i}.out),'LineWidth',1);
    end
   
    if kk==1
        title('Output projection over main PCs')
    end
    ylabel(['PC ' num2str(kk)])
end

% Plot STATES
for i=1:M
subplot(Nx,Ny,Ny*i) %States
hold on
%Smooth states
set(gca,'ColorOrderIndex',1)
clear p
for k=1:length(model)
p(k)=plot(model{k}.smoothStates(i,:),'LineWidth',2,'DisplayName',[model{k}.name ', \tau=' num2str(-1./log(model{k}.J(i,i)),3)]);
end
%axis([0 2000 -.5 1.5])
%MLE states:
set(gca,'ColorOrderIndex',1)
clear p
for k=1:length(model)
    p(k)=plot(model{k}.X(i,:),'LineWidth',1,'DisplayName',[model{k}.name ', \tau=' num2str(-1./log(model{k}.J(i,i)),3)]);
    try
    patch([1:size(model{k}.X,2),size(model{k}.X,2):-1:1]',[model{k}.X(i,:)+sqrt(squeeze(model{k}.P(i,i,:)))', fliplr(model{k}.X(i,:)-sqrt(squeeze(model{k}.P(i,i,:)))')]',p(k).Color,'EdgeColor','none','FaceAlpha',.3)
    end
end
title('States')
legend(p,'Location','SouthEast')
ylabel(['State ' num2str(i)])
end

%Smooth output RMSE
subplot(Nx,Ny,Ny*(M+1)) 
hold on
for k=1:length(model)
aux1=sqrt(sum((Y-model{k}.smoothOut).^2));
p1=plot(aux1,'LineWidth',1);
bar2=bar([1600+k*100],mean([aux1]),'EdgeColor','none','BarWidth',100,'FaceColor',p1.Color);
text(1700+(k-1)*100,.8+k*.2,['LogL=' num2str(model{k}.logL)],'Color',bar2.FaceColor)
end
title('Smooth output error (RMSE)')
axis([0 2200 .0 1.5])
grid on

%MLE state output error
subplot(Nx,Ny,Ny*(M+2)) 
hold on
for k=1:length(model)
aux1=sqrt(sum((Y-model{k}.out).^2));
p1=plot(aux1,'LineWidth',1);
bar2=bar([1600+k*100],mean([aux1]),'EdgeColor','none','BarWidth',100,'FaceColor',p1.Color);
text(1700+(k-1)*100,.8+.2*k,['LogL=' num2str(model{k}.logL)],'Color',bar2.FaceColor)
end
title('Output error (RMSE)')
axis([0 2200 .0 1.5])
grid on

 %One ahead error
subplot(Nx,Ny,Ny*(M+3)-1)
hold on
for k=1:length(model)
aux1=sqrt(sum((Y(:,2:end)-model{k}.oneAheadOut).^2));
p1=plot(aux1,'LineWidth',1);
bar2=bar([1600+k*100],mean([aux1]),'EdgeColor','none','BarWidth',100,'FaceColor',p1.Color);
text(1700+(k-1)*100,.8+k*.3,['LogL=' num2str(model{k}.logL)],'Color',bar2.FaceColor)
end
title('One-ahead output error (RMSE)')
axis tight
grid on

%MLE state innovation
subplot(Nx,Ny,Ny*(M+3)) 
hold on
for k=1:length(model)
aux1=sqrt(sum((model{k}.X(:,2:end)-model{k}.oneAheadStates).^2));
p1=plot(aux1,'LineWidth',1);
bar2=bar([1600+k*100],mean([aux1]),'EdgeColor','none','BarWidth',100,'FaceColor',p1.Color);
text(1700+(k-1)*100,.02+.02*k,['LogL=' num2str(model{k}.logL)],'Color',bar2.FaceColor)
end
title('One-ahead state error (RMSE)')
axis tight
grid on

% Plot features of one-ahead output residuals:
for i=1:length(model)
    res=Y(:,2:end)-model{i}.oneAheadOut;
    [pp,cc,aa]=pca((res)','Centered','off');
    subplot(Nx,ceil(3*Ny/2),(kk+1)*ceil(3*Ny/2)+1)
    hold on
    p(i)=plot(cc(:,1),'LineWidth',1) ;
    title('First PC of residual')
    
    subplot(Nx,ceil(3*Ny/2),(kk+1)*ceil(3*Ny/2)+2)
    hold on
    qq1=qqplot(cc(:,1));
    qq1(1).MarkerEdgeColor=p(i).Color;
    ax=gca;
    ax.Title.String='QQ plot residual PC 1';
    
    subplot(Nx,ceil(3*Ny/2),(kk+1)*ceil(3*Ny/2)+3)
    hold on
    r=fftshift(xcorr(cc(:,1)));
    plot(0:length(r)-1,r)
    axis tight
    aa=axis;
    grid on
    xlabel('Delay (samp)')
    title('Residual PC 1 autocorr')
    axis([0 20 aa(3:4)])
end