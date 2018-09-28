function [fh,fh2] = vizDataFit(model,Y,U)

M=size(model{1}.X,1);
fh=figure('Units','Normalized','OuterPosition',[0 0 1 1]);
Ny=3;
Nx=max(M,6);
yoff=size(Y,2)*1.1;
%% Compute output and residuals
for i=1:length(model)
    if ~isfield(model{i},'P')
        model{i}.P0=[];
    else
        model{i}.P0=model{i}.P(:,:,1);
    end
    fastFlag=0;
    [Xs,Ps,Pt,Xf,Pf,Xp,Pp,rejSamples]=statKalmanSmoother(Y,model{i}.J,model{i}.C,model{i}.Q,model{i}.R,[],[],model{i}.B,model{i}.D,U,false,fastFlag); 
    model{i}.Xs=Xs; %Smoothed data
    model{i}.Pp=Pp; %One-step ahead uncertainty from filtered data.
    model{i}.Pf=Pf;
    model{i}.Xf=Xf; %Filtered data
    model{i}.Xp=Xp; %Predicted data
    model{i}.out=model{i}.C*model{i}.Xs+model{i}.D*U;
    model{i}.res=Y-model{i}.out;
    model{i}.oneAheadStates=model{i}.J*model{i}.Xs(:,1:end-1)+model{i}.B*U(:,1:end-1);
    model{i}.oneAheadOut=model{i}.C*(model{i}.oneAheadStates)+model{i}.D*U(:,2:end);
    [Y2,X2]=fwdSim(U,model{i}.J,model{i}.B,model{i}.C,model{i}.D,model{i}.Xs(:,1),[],[]); %Simulating from MLE initial state
    model{i}.smoothStates=X2;
    model{i}.smoothOut=Y2;

    model{i}.logLtest=dataLogLikelihood(Y,U,model{i}.J,model{i}.B,model{i}.C,model{i}.D,model{i}.Q,model{i}.R,[],[],'approx');
end
%% Define colormap:
ex1=[1,0,0];
ex2=[0,0,1];
mid=ones(1,3);
N=100;
map=[ex1.*[N:-1:1]'/N + mid.*[0:N-1]'/N; mid; ex2.*[0:N-1]'/N + mid.*[N:-1:1]'/N];

%% Plot output PCs and residuals
[cc,pp,aa]=pca(Y','Centered','off');
maxK=min(5,size(Y,1));
cc=cc(:,1:maxK);
for kk=1:maxK
    subplot(Nx,3*Ny,3*(kk-1)*Ny+1) %PC of true data
    hold on
    Nc=size(cc,1);
    try
        imagesc(flipud(reshape(cc(:,kk),12,Nc/12)'))
    catch
        imagesc(flpiud(cc(:,kk)))
    end
    colormap(flipud(map))
    aC=max(abs(cc(:)));
    caxis([-aC aC])
    axis tight
    subplot(Nx,3*Ny,3*(kk-1)*Ny+[2:3]) %Output along PC of true data
    hold on
    scatter(1:size(Y,2),cc(:,kk)'*Y,5,'filled','k')
     set(gca,'ColorOrderIndex',1)
    for i=1:length(model)
        p(i)=plot(cc(:,kk)'*(model{i}.out),'LineWidth',2);
    end
        
    if kk==1
        title('Output projection over main PCs')
    end
    ylabel(['PC ' num2str(kk)])
end
%% Measures of output error:
%Smooth output RMSE
subplot(Nx,Ny,2) 
hold on
for k=1:length(model)
aux1=sqrt(sum((Y-model{k}.smoothOut).^2));
p1=plot(aux1,'LineWidth',1);
bar2=bar([yoff+k*100],mean([aux1]),'EdgeColor','none','BarWidth',100,'FaceColor',p1.Color);
text(yoff+(k)*100-50,mean([aux1])*(1+k*.5),[num2str(nanmean(aux1))],'Color',bar2.FaceColor)
end
title('Smooth output error (RMSE)')
axis tight
grid on
set(gca,'YScale','log')

% MLE state output error
subplot(Nx,Ny,Ny+2) 
hold on
for k=1:length(model)
aux1=sqrt(sum((Y-model{k}.out).^2));
p1=plot(aux1,'LineWidth',1);
bar2=bar([yoff+k*100],mean([aux1]),'EdgeColor','none','BarWidth',100,'FaceColor',p1.Color);
text(yoff+(k)*100-50,mean([aux1])*(1+k*.5),['logL=' num2str(model{k}.logLtest,6)],'Color',bar2.FaceColor)
end
title('MLE-state output error (RMSE)')
axis tight
grid on
set(gca,'YScale','log')

%One ahead error
subplot(Nx,Ny,2*Ny+2)
hold on
for k=1:length(model)
aux1=sqrt(sum((Y(:,2:end)-model{k}.oneAheadOut).^2));
p1=plot(aux1,'LineWidth',1);
bar2=bar([yoff+k*100],nanmean([aux1]),'EdgeColor','none','BarWidth',100,'FaceColor',p1.Color);
text(yoff+(k)*100-50,nanmean(aux1)*(1+.5*k),[num2str(nanmean(aux1))],'Color',bar2.FaceColor)
end
title('MLE one-ahead output error (RMSE)')
axis tight
grid on
set(gca,'YScale','log')

%% Plot STATES
clear p
for k=1:length(model)
for i=1:M
subplot(Nx,Ny,Ny*(i-1)+3) %States
hold on
%Smooth states
set(gca,'ColorOrderIndex',k)
p(k,i)=plot(model{k}.smoothStates(i,:),'LineWidth',2,'DisplayName',[model{k}.name ', \tau=' num2str(-1./log(model{k}.J(i,i)),3)]);
%MLE states:
plot(model{k}.Xf(i,:),'LineWidth',1,'DisplayName',[model{k}.name ', \tau=' num2str(-1./log(model{k}.J(i,i)),3)],'Color',p(k).Color);
patch([1:size(model{k}.Xf,2),size(model{k}.Xf,2):-1:1]',[model{k}.Xf(i,:)+sqrt(squeeze(model{k}.Pf(i,i,:)))', fliplr(model{k}.Xf(i,:)-sqrt(squeeze(model{k}.Pf(i,i,:)))')]',p(k).Color,'EdgeColor','none','FaceAlpha',.3)
if k==length(model)
    legend(p(:,i),'Location','SouthEast')
    title('(Filtered) States')
    ylabel(['State ' num2str(i)])
end
end
end

%% MLE state innovation
subplot(Nx,Ny,Ny*(M)+3) 
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

%% Plot features of one-ahead output residuals:
Nny=ceil(4);
for i=1:length(model)
    res=Y(:,2:end)-model{i}.oneAheadOut; %One-ahead residuals
    
    
    [~,cc,~]=pca((res)','Centered','off');
    subplot(Nx,Nny,(Nx-1)*Nny+1)
    hold on
    p(i)=plot(cc(:,1),'LineWidth',1) ;
    title('First PC of residual')
    
    subplot(Nx,Nny,(Nx-1)*Nny+2)
    hold on
    qq1=qqplot(cc(:,1));
    qq1(1).MarkerEdgeColor=p(i).Color;
    ax=gca;
    ax.Title.String='QQ plot residual PC 1';
    
    subplot(Nx,Nny,(Nx-1)*Nny+3)
    hold on
    r=xcorr(cc(:,1));
    plot(-(length(r)-1)/2:(length(r)-1)/2,r)
    axis tight
    aa=axis;
    grid on
    xlabel('Delay (samp)')
    title('Residual PC 1 autocorr')
    axis([-15 15 aa(3:4)])
    
    subplot(Nx,Nny,(Nx-1)*Nny+4)
    hold on
    histogram(cc(:,1),'EdgeColor','none','Normalization','pdf','FaceAlpha',.2,'BinEdges',[-1:.02:1])
end
xx=[-1:.001:1];
subplot(Nx,Nny,(Nx-1)*Nny+4)
hold on
sig=.25;
plot(xx,exp(-(xx.^2)/(2*sig^2))/sqrt(2*pi*sig^2),'k')
title('Residual PC 1 histogram')
  
%% Compare outputs at different points in time:
fh=figure('Units','Normalized','OuterPosition',[0 0 1 1]);
N=size(Y,2);
viewPoints=[1,40,51,151,251,651,940,951,1001,1101,N-11]+5;
binw=10;
viewPoints(viewPoints>N-binw/2)=[];
Ny=length(viewPoints);
M=length(model);
aC=max(abs(Y(:)));
for i=1:Ny
    subplot(M+1,Ny,i)
    try
        imagesc(reshape(mean(Y(:,viewPoints(i)+[-(binw/2):(binw/2)]),2),12,size(Y,1)/12)')
    catch
        imagesc(mean(Y(:,viewPoints(i)+[-(binw/2):(binw/2)]),2))
    end
    colormap(flipud(map))
    caxis([-aC aC])
    axis tight
    title(['Output at t=' num2str(viewPoints(i))])
    if i==1
        ylabel(['Data, binwidth=' num2str(binw)])
    end
    for k=1:M
       subplot(M+1,Ny,i+k*Ny)
        try
            imagesc(reshape(mean(model{k}.out(:,viewPoints(i)+[-(binw/2):(binw/2)]),2),12,size(Y,1)/12)')
        catch
            imagesc(mean(model{k}.out(:,viewPoints(i)+[-(binw/2):(binw/2)]),2))
        end
        colormap(flipud(map))
        caxis([-aC aC])
        axis tight
        if i==1
            ylabel(model{k}.name)
        end
    end
end