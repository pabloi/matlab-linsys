function [fh] = compareModels(model,Y,U)

M=size(model{1}.X,1);
fh=figure;
Ny=4;
Nx=M+4;
yoff=size(Y,2)*1.1;
%Compute output and residuals
for i=1:length(model)
    if ~isfield(model{i},'P')
        model{i}.P0=[];
    else
        model{i}.P0=model{i}.P(:,:,1);
    end
    fastFlag=0;
    [Xs,Ps,Pt,Xf,Pf,Xp,Pp,rejSamples]=statKalmanSmoother(Y,model{i}.J,model{i}.C,model{i}.Q,model{i}.R,model{i}.X(:,1),model{i}.P0,model{i}.B,model{i}.D,U,false,fastFlag); 
    model{i}.Xs=Xs; %Smoothed data
    model{i}.Pp=Pp; %One-step ahead uncertainty from filtered data.
    model{i}.Pf=Pf;
    model{i}.Xf=Xf; %Filtered data
    model{i}.Xp=Xp; %Predicted data
    model{i}.out=model{i}.C*model{i}.Xs+model{i}.D*U;
    model{i}.res=Y-model{i}.out;
    model{i}.oneAheadStates=model{i}.J*model{i}.Xs(:,1:end-1)+model{i}.B*U(:,1:end-1);
    model{i}.oneAheadOut=model{i}.C*(model{i}.oneAheadStates)+model{i}.D*U(:,2:end);
    [Y2,X2]=fwdSim(U,model{i}.J,model{i}.B,model{i}.C,model{i}.D,model{i}.X(:,1),[],[]);
    model{i}.smoothStates=X2;
    model{i}.smoothOut=Y2;

    model{i}.logLtest=dataLogLikelihood(Y,U,model{i}.J,model{i}.B,model{i}.C,model{i}.D,model{i}.Q,model{i}.R,model{i}.X(:,1),model{i}.P0,'approx');
end

% Plot output PCs and residuals
[cc,pp,aa]=pca(Y','Centered','off');
for kk=1:min(M+2,size(Y,1))
    subplot(Nx,Ny,(kk-1)*Ny+1) %Output along first PC of true data
    hold on
scatter(1:size(Y,2),cc(:,kk)'*Y,5,'filled','k')
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
subplot(Nx,Ny,Ny*(i-1)+2) %States
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
    p(k)=plot(model{k}.Xf(i,:),'LineWidth',1,'DisplayName',[model{k}.name ', \tau=' num2str(-1./log(model{k}.J(i,i)),3)]);
    patch([1:size(model{k}.Xf,2),size(model{k}.Xf,2):-1:1]',[model{k}.Xf(i,:)+sqrt(squeeze(model{k}.Pf(i,i,:)))', fliplr(model{k}.Xf(i,:)-sqrt(squeeze(model{k}.Pf(i,i,:)))')]',p(k).Color,'EdgeColor','none','FaceAlpha',.3)
end
title('(Filtered) States')
%if i==1
legend(p,'Location','SouthEast')
%end
ylabel(['State ' num2str(i)])
end

%Smooth output RMSE
subplot(Nx,Ny,Ny*(M+2)+1) 
hold on
for k=1:length(model)
aux1=sqrt(sum((Y-model{k}.smoothOut).^2));
p1=plot(aux1,'LineWidth',1);
bar2=bar([yoff+k*100],mean([aux1]),'EdgeColor','none','BarWidth',100,'FaceColor',p1.Color);
text(yoff+(k)*100,.8+k*.2,['LogL=' num2str(model{k}.logLtest)],'Color',bar2.FaceColor)
end
title('Smooth output error (RMSE)')
axis tight
grid on

%MLE state output error
subplot(Nx,Ny,Ny*(M+1)+2) 
hold on
for k=1:length(model)
aux1=sqrt(sum((Y-model{k}.out).^2));
p1=plot(aux1,'LineWidth',1);
bar2=bar([yoff+k*100],mean([aux1]),'EdgeColor','none','BarWidth',100,'FaceColor',p1.Color);
text(yoff+(k)*100,.8+.2*k,['LogL=' num2str(model{k}.logLtest)],'Color',bar2.FaceColor)
end
title('Output error (RMSE)')
axis tight
grid on

 %One ahead error
subplot(Nx,Ny,Ny*(M+2)+2)
hold on
for k=1:length(model)
aux1=sqrt(sum((Y(:,2:end)-model{k}.oneAheadOut).^2));
p1=plot(aux1,'LineWidth',1);
bar2=bar([yoff+k*100],nanmean([aux1]),'EdgeColor','none','BarWidth',100,'FaceColor',p1.Color);
text(yoff+(k)*100,nanmean(aux1)*(1+.5*k),[num2str(nanmean(aux1))],'Color',bar2.FaceColor)
end
title('One-ahead output error (RMSE)')
axis tight
grid on

%MLE state innovation
subplot(Nx,Ny,Ny*(M)+2) 
hold on
for k=1:length(model)
    stError=model{k}.Xs(:,2:end)-model{k}.oneAheadStates;
aux1=sqrt(z2score(stError,model{k}.Q));
p1=plot(aux1,'LineWidth',1);
bar2=bar([yoff + k*100],nanmean([aux1]),'EdgeColor','none','BarWidth',100,'FaceColor',p1.Color);
text(yoff+(k)*100,nanmean(aux1)*(1+.5*k),[num2str(nanmean(aux1))],'Color',bar2.FaceColor)
end
title('(Smoothed) One-ahead state error (z-score)')
axis tight
grid on

% Plot features of one-ahead output residuals:
Nny=ceil(4*Ny/2);
for i=1:length(model)
    res=Y(:,2:end)-model{i}.oneAheadOut; %One-ahead residuals
    
    
    [~,cc,~]=pca((res)','Centered','off');
    subplot(Nx,Nny,(kk+1)*Nny+1)
    hold on
    p(i)=plot(cc(:,1),'LineWidth',1) ;
    title('First PC of residual')
    
    subplot(Nx,Nny,(kk+1)*Nny+2)
    hold on
    qq1=qqplot(cc(:,1));
    qq1(1).MarkerEdgeColor=p(i).Color;
    ax=gca;
    ax.Title.String='QQ plot residual PC 1';
    
    subplot(Nx,Nny,(kk+1)*Nny+3)
    hold on
    r=xcorr(cc(:,1));
    plot(-(length(r)-1)/2:(length(r)-1)/2,r)
    axis tight
    aa=axis;
    grid on
    xlabel('Delay (samp)')
    title('Residual PC 1 autocorr')
    axis([-15 15 aa(3:4)])
    
    subplot(Nx,Nny,(kk+1)*Nny+4)
    hold on
    histogram(cc(:,1),'EdgeColor','none','Normalization','pdf','FaceAlpha',.2,'BinEdges',[-1:.02:1])
end
xx=[-1:.001:1];
    subplot(Nx,Nny,(kk+1)*Nny+4)
    hold on
    sig=.25;
    plot(xx,exp(-(xx.^2)/(2*sig^2))/sqrt(2*pi*sig^2),'k')
    title('Residual PC 1 histogram')