function [fh,fh2] = vizDataFit(model,Y,U)

M=max(cellfun(@(x) size(x.J,1),model));
fh=figure('Units','Normalized','OuterPosition',[0 0 1 1]);
Ny=3;
Nx=max(M+1,6);
yoff=size(Y,2)*1.1;
Nm=length(model); %Number of models
%% Add reduced models:
for i=1:length(model)
%    redModel{i}=model{i};
%    [redModel{i}.C,redModel{i}.R,redModel{i}.D,redModel{i}.Y_]=reduceModel(model{i}.C,model{i}.R,model{i}.D,Y);
    model{i}.Y_=Y;
end
%% Compute output and residuals
modelOrig=model;
for k=1%:2 %Original or reduced models
    if k==1
        model=modelOrig;
    else
        model=redModel;
    end
    for i=1:length(model)
        %[model{i}.J,model{i}.B,model{i}.C,~,~,model{i}.Q] = canonize(model{i}.J,model{i}.B,model{i}.C,[],model{i}.Q,[],[]);
        opts.fastFlag=false;
        Nd=size(model{i}.D,2);
        [Xs,Ps,Pt,Xf,Pf,Xp,Pp,rejSamples,logL]=statKalmanSmoother(model{i}.Y_,model{i}.J,model{i}.C,model{i}.Q,model{i}.R,[],[],model{i}.B,model{i}.D,U,opts);
        model{i}.Xs=Xs; %Smoothed data
        model{i}.Ps=Ps;
        model{i}.Pp=Pp; %One-step ahead uncertainty from filtered data.
        model{i}.Pf=Pf;
        model{i}.Xf=Xf; %Filtered data
        model{i}.Xp=Xp; %Predicted data
        model{i}.out=model{i}.C*model{i}.Xs+model{i}.D*U(1:Nd,:); %Discarding input components at the end
        model{i}.outF=model{i}.C*model{i}.Xf+model{i}.D*U(1:Nd,:); %Discarding input components at the end
        model{i}.res=model{i}.Y_-model{i}.out;
        model{i}.oneAheadStates=model{i}.J*model{i}.Xs(:,1:end-1)+model{i}.B*U(1:Nd,1:end-1);
        model{i}.oneAheadOut=model{i}.C*(model{i}.oneAheadStates)+model{i}.D*U(1:Nd,2:end);
        model{i}.oneAheadOutF=model{i}.C*(model{i}.Xp(:,2:end-1))+model{i}.D*U(1:Nd,2:end);
        [Y2,X2]=fwdSim(U(1:Nd,:),model{i}.J,model{i}.B,model{i}.C,model{i}.D,model{i}.Xs(:,1),[],[]); %Simulating from MLE initial state
        model{i}.smoothStates=X2;
        model{i}.smoothOut=Y2;
        model{i}.logLtest=logL;
        [bic,aic,bic2]= bicaic(model{i},model{i}.Y_,numel(Y)*model{i}.logLtest);
        model{i}.BIC=bic/(2*numel(Y)); %To put in the same scale as logL
        model{i}.BIC2=bic2/(2*numel(Y)); %To put in the same scale as logL
        model{i}.AIC=aic/(2*numel(Y));
    end
    if k==1
        modelOrig=model;
    else
        redModel=model;
    end
end
model=modelOrig;
%% Define colormap:
ex1=[1,0,0];
ex2=[0,0,1];
mid=ones(1,3);
N=100;
map=[ex1.*[N:-1:1]'/N + mid.*[0:N-1]'/N; mid; ex2.*[0:N-1]'/N + mid.*[N:-1:1]'/N];

%% Plot output PCs and fir
[cc,pp,aa]=pca(Y','Centered',false);
maxK=min(5,size(Y,1));
cc=cc(:,1:maxK);
for kk=1:maxK
    subplot(Nx,3*Ny,3*(kk-1)*Ny+1) %PC of true data
    hold on
    Nc=size(cc,1);
    try
        imagesc(flipud(reshape(cc(:,kk),12,Nc/12)'))
    catch
        imagesc(flipud(cc(:,kk)))
    end
    colormap(flipud(map))
    aC=max(abs(cc(:)));
    caxis([-aC aC])
    axis tight
    subplot(Nx,3*Ny,3*(kk-1)*Ny+[2:3]) %Output along PC of true data
    hold on
    scatter(1:size(Y,2),cc(:,kk)'*Y,5,.5*ones(1,3),'filled')
     set(gca,'ColorOrderIndex',1)
    for i=1:length(model)
        p(i)=plot(cc(:,kk)'*(model{i}.out),'LineWidth',2);
    end

    if kk==1
        title('Output projection over main PCs')
    end
    ylabel(['PC ' num2str(kk) ', ' num2str(100*aa(kk)/sum(aa)) '%'])
end

%% Measures of output error:
binw=11;
for ll=1:3
    subplot(Nx,Ny,2+(ll-1)*Ny)
    hold on
    for k=1:length(model)
        switch ll
            case 1 %Smooth output RMSE
                aux1=sqrt(sum((Y-model{k}.smoothOut).^2));
                title('Smooth output error (RMSE, mov. avg.)')
            case 2 % MLE state output error
                aux1=sqrt(sum((Y(:,2:end)-model{k}.oneAheadOut).^2));
                title('KS one-ahead output error (RMSE, mov. avg.)')
            case 3 %One ahead error
                aux1=sqrt(sum((Y(:,2:end)-model{k}.oneAheadOutF).^2));
                title('KF prediction output error (RMSE, mov. avg.)')
        end
        idx=find(~isnan(aux1));
        aux1=aux1(idx);
        aux2=conv(aux1,ones(1,binw)/binw,'valid');
        p1=plot(idx((binw-1)/2+1:end-(binw-1)/2),aux2,'LineWidth',1);
        RMSE=sqrt(mean([aux1].^2)); %Normalized Frobenius norm
        bar2=bar([yoff+k*200],RMSE,'EdgeColor','none','BarWidth',200,'FaceColor',p1.Color);
        text(yoff+(k-1)*200+100,.9*RMSE,[num2str(RMSE,4)],'Color','w','FontSize',7)
    end
    axis tight
    grid on
    set(gca,'YScale','log')
end

%LogL, BIC, AIC
for kl=1%:2
    switch kl
        case 1
            mdl=model;
        case 2
            mdl=redModel; %To do: used reduced model
    end
    for kj=1:3 %logL, BIC, aic
        switch kj
            case 1
                yy=cellfun(@(x) x.logLtest,mdl);
                nn='logL';
            case 2
                yy=cellfun(@(x) x.BIC,mdl);
                nn='-BIC';
            case 3
                yy=cellfun(@(x) x.AIC,mdl);
                nn='-AIC';
        end
        subplot(Nx,3*Ny,9*Ny+3+kj+(kl-1)*3*Ny)
        hold on
        [N,Nz]=size(Y);
        Mm=length(mdl);
        for k=1:Mm
            set(gca,'ColorOrderIndex',k)
            bar2=bar([k*100],yy(k),'EdgeColor','none','BarWidth',100);
            text((k)*100,.982*(yy(k)),[num2str(yy(k),6)],'Color','w','FontSize',8,'Rotation',90)
        end
        title([nn])
        grid on
        %set(gca,'XTick',100*(Mm+1)*.5,'XTickLabel',nn)
        axis tight;
        aa=axis;
        axis([aa(1:2) .98*min(yy) max(yy)])
    end
end

%% Plot STATES
clear p
for k=1:length(model)
    taus=-1./log(sort(eig(model{k}.J)));
    for i=1:size(model{k}.J,1)
        subplot(Nx,Ny,Ny*(i-1)+3) %States
        hold on
        if i==1
            nn=[model{k}.name ', \tau=' num2str(taus(i),3)];
        else
            nn=['\tau=' num2str(taus(i),3)];
        end
        %Smooth states
        set(gca,'ColorOrderIndex',k)
        p(k,i)=plot(model{k}.smoothStates(i,:),'LineWidth',2,'DisplayName',nn);
        %MLE states:
        %plot(model{k}.Xf(i,:),'LineWidth',1,'DisplayName',nn,'Color',p(k).Color);
        %patch([1:size(model{k}.Xf,2),size(model{k}.Xf,2):-1:1]',[model{k}.Xf(i,:)+sqrt(squeeze(model{k}.Pf(i,i,:)))', %fliplr(model{k}.Xf(i,:)-sqrt(squeeze(model{k}.Pf(i,i,:)))')]',p(k).Color,'EdgeColor','none','FaceAlpha',.3)
        plot(model{k}.Xs(i,:),'LineWidth',1,'DisplayName',nn,'Color',p(k).Color);
        patch([1:size(model{k}.Xs,2),size(model{k}.Xs,2):-1:1]',[model{k}.Xs(i,:)+sqrt(squeeze(model{k}.Ps(i,i,:)))', fliplr(model{k}.Xs(i,:)-sqrt(squeeze(model{k}.Ps(i,i,:)))')]',p(k).Color,'EdgeColor','none','FaceAlpha',.3)
        %plot(model{k}.Xp(i,:),'LineWidth',1,'DisplayName',nn,'Color','k');
        if k==length(model)
            legend(findobj(gca,'Type','Line','LineWidth',2),'Location','SouthEast')
            title('(Smoothed) States')
            ylabel(['State ' num2str(i)])
        end
        axis tight
        aa=axis;
        axis([aa(1:2) -.3 1.3]);
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
text(yoff+(k)*100,nanmean(aux1)*(1+.3*k),[num2str(nanmean(aux1))],'Color',bar2.FaceColor)
end
title('(KS) Predicted state error (z-score)')
axis tight
grid on

%% Plot features of one-ahead output residuals:
Nny=6;
allLogL=cellfun(@(x) x.logLtest,model);
for i=1:length(model)
    res=Y(:,2:end)-model{i}.oneAheadOut; %One-ahead residuals
    res=substituteNaNs(res'); %Removing NaNs, otherwise this is all crap
    [pp,cc,aa]=pca((res),'Centered','off');
    subplot(Nx,Nny,(Nx-1)*Nny+1)
    if model{i}.logLtest==min(allLogL)
        try
            imagesc(flipud(reshape(pp(:,kk),12,Nc/12)'))
        catch
            imagesc(flipud(pp(:,kk)))
        end
        colormap(flipud(map))
        aC=.5*max(abs(cc(:)));
        caxis([-aC aC])
        title('First residual PC for best model')
    end


    subplot(Nx,Nny,(Nx-1)*Nny+2)
    hold on
    aux1=conv(cc(:,1)',ones(1,binw)/binw,'valid');
    p(i)=plot(aux1,'LineWidth',1) ;
    title('First PC of residual, mov. avg.')
    grid on


    subplot(Nx,Nny,(Nx-1)*Nny+3)
    hold on
    plot(aa,'LineWidth',1)
    title('Distribution of residual energy')
    grid on
    set(gca,'YScale','log')
    axis([1 30 1e-2 1])

    subplot(Nx,Nny,(Nx-1)*Nny+4)
    hold on
    qq1=qqplot(cc(:,1));
    qq1(1).MarkerEdgeColor=p(i).Color;
    ax=gca;
    ax.Title.String='QQ plot residual PC 1';

    subplot(Nx,Nny,(Nx-1)*Nny+5)
    hold on
    r=xcorr(cc(:,1));
    plot(-(length(r)-1)/2:(length(r)-1)/2,r)
    axis tight
    aa=axis;
    grid on
    xlabel('Delay (samp)')
    title('Residual PC 1 autocorr')
    axis([-15 15 aa(3:4)])

    subplot(Nx,Nny,(Nx-1)*Nny+6)
    hold on
    histogram(cc(:,1),'EdgeColor','none','Normalization','pdf','FaceAlpha',.2,'BinEdges',[-1:.02:1])
end
xx=[-1:.001:1];
subplot(Nx,Nny,(Nx-1)*Nny+6)
hold on
sig=.25;
plot(xx,exp(-(xx.^2)/(2*sig^2))/sqrt(2*pi*sig^2),'k')
title('Residual PC 1 histogram')

%% Compare outputs at different points in time:
if nargout>1
fh2=figure('Units','Normalized','OuterPosition',[0 0 1 1]);
N=size(Y,2);
viewPoints=[1,40,51,151,251,651,940,951,1001,1101,N-11]+5;
binw=10;
viewPoints(viewPoints>N-binw/2)=[];
Ny=length(viewPoints);
M=length(model);
aC=max(abs(Y(:)));
for i=1:Ny
    subplot(M+1,Ny,i)
    trueD=Y(:,viewPoints(i)+[-(binw/2):(binw/2)]);
    try
        imagesc(reshape(nanmean(trueD,2),12,size(Y,1)/12)')
    catch
        imagesc(nanmean(trueD,2))
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
       dd=trueD-model{k}.out(:,viewPoints(i)+[-(binw/2):(binw/2)]);
        try
            imagesc(reshape(nanmean(dd,2),12,size(Y,1)/12)')
        catch
            imagesc(nanmean(dd,2))
        end
        colormap(flipud(map))
        caxis([-aC aC])
        axis tight
        mD=sqrt(mean(sum(trueD.^2),2));
        mD=1;
        title(['RMSE=' num2str(sqrt(nanmean(sum((dd).^2),2))/mD)])
        if i==1
            ylabel(model{k}.name)
        end
    end
end
end
