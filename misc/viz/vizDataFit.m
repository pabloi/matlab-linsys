function [fh,fh2] = vizDataFit(model,Y,U)
  if ~iscell(model)
    model={model};
  end
  if isa(model{1},'struct') %For back-compatibility
    model=cellfun(@(x) linsys.struct2linsys(x),model,'UniformOutput',false);
  end
  if nargin<3
    datSet=Y; %Expecting dset object
  else
    datSet=dset(U,Y);
  end
  Y=datSet.out;
  U=datSet.in;
M=max(cellfun(@(x) size(x.A,1),model(:)));
fh=figure('Units','Normalized','OuterPosition',[0 0 1 1]);
Ny=4;
Nx=max(M,6);
yoff=size(Y,2)*1.1;
Nm=length(model); %Number of models

for i=1:length(model)
  dFit{i}=model{i}.fit(datSet);
end

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
    ylabel(['PC ' num2str(kk) ', ' num2str(100*aa(kk)/sum(aa)) '%'])
    axis tight
    subplot(Nx,3*Ny,3*(kk-1)*Ny+[2:3]) %Output along PC of true data
    hold on
    scatter(1:size(Y,2),cc(:,kk)'*Y,5,.5*ones(1,3),'filled')
     set(gca,'ColorOrderIndex',1)
    for i=1:length(model)
        [~,modelOut]=datSet.getOneAheadResiduals(dFit{i});
        p(i)=plot(cc(:,kk)'*(modelOut),'LineWidth',2);
    end

    if kk==1
        title('One-ahead (KF) output projected onto data PCs')
    end
    axis tight
end

%% Measures of output error:
binw=11;
for ll=1:2
    for k=1:length(model)
        switch ll
            case 1 %Deterministic output RMSE
                iC=dFit{k}.MLEstate.getSample(1); %MLE estimate of init cond
                [simSet]=model{k}.simulate(U,iC,true);
                res=datSet.out -simSet.out;
                aux1=sqrt(sum((res).^2));
                tt=('Deterministic output error (RMSE, mov. avg.)');
            case 3 % MLE state output error
                tt=('KS one-ahead output error (RMSE, mov. avg.)');
            case 2 %One ahead error
                res=datSet.getOneAheadResiduals(dFit{k});
                aux1=sqrt(sum((res).^2));
                tt=('KF prediction output error (RMSE, mov. avg.)');
        end
        idx=find(~isnan(aux1));
        aux2=aux1(idx);
        aux2=conv(aux2,ones(1,binw)/binw,'valid');
        subplot(Nx,Ny,2+(2*ll-2)*Ny) %Time-course of residuals
        hold on
        set(gca,'ColorOrderIndex',k)
        s1=scatter(1:length(aux1),aux1,5,'filled','MarkerEdgeColor','none','MarkerFaceAlpha',.5);
        %p1=plot(idx((binw-1)/2+1:end-(binw-1)/2),aux2,'LineWidth',1,'LineColor',s1.MarkerFaceColor);
        if k==1
            title(tt)
            axis tight
            grid on
            set(gca,'YScale','log')
        end
        subplot(Nx,Ny,2+(2*ll-1)*Ny) %Bars of residuals
        hold on
        RMSE=sqrt(mean([aux1].^2)); %Normalized Frobenius norm
        bar2=bar(k,RMSE,'EdgeColor','none','BarWidth',1,'FaceColor',s1.MarkerFaceColor);
        text(k,.9*RMSE,[num2str(RMSE,4)],'Color','w','FontSize',6,'Rotation',270)
        set(gca,'XTick',1:length(model))
         if k==length(model)
            axis tight
            grid on
            set(gca,'YScale','log')
        end
    end

end

%% Plot STATES
clear p
for k=1:length(model)
    taus=-1./log(sort(eig(model{k}.A)));
    [projectedX,projectedXLS]=getDataProjections(datSet,model{k});
    Xs=dFit{k}.MLEstate.state;
    %dXs=Xs(:,2:end)-model{k}.A*Xs(:,1:end-1)-model{k}.B*U(:,1:end-1);
    Ps=dFit{k}.MLEstate.covar;
    iC=dFit{k}.MLEstate.getSample(1); %MLE estimate of init cond
    [simSet,simState]=model{k}.simulate(U,iC,true);
    Xsim=simState.state;
    Psim=simState.covar;
    for i=1:size(model{k}.A,1)
        subplot(Nx,Ny,Ny*(i-1)+3) %States
        hold on
        if i==1
            nn=[model{k}.name ', \tau=' num2str(taus(i),3)];
        else
            nn=['\tau=' num2str(taus(i),3)];
        end
        %Smooth states
        set(gca,'ColorOrderIndex',k)
        %MLE states:
        %plot(model{k}.Xf(i,:),'LineWidth',1,'DisplayName',nn,'Color',p(k).Color);
        %patch([1:size(model{k}.Xf,2),size(model{k}.Xf,2):-1:1]',[model{k}.Xf(i,:)+sqrt(squeeze(model{k}.Pf(i,i,:)))', %fliplr(model{k}.Xf(i,:)-sqrt(squeeze(model{k}.Pf(i,i,:)))')]',p(k).Color,'EdgeColor','none','FaceAlpha',.3)
        p(k,i)=plot(Xs(i,:),'LineWidth',2,'DisplayName',nn);
        %plot(model{k}.smoothStates(i,:),'LineWidth',1,'DisplayName',nn);
        patch([1:size(Xs,2),size(Xs,2):-1:1]',[Xs(i,:)+sqrt(squeeze(Ps(i,i,:)))', fliplr(Xs(i,:)-sqrt(squeeze(Ps(i,i,:)))')]',p(k).Color,'EdgeColor','none','FaceAlpha',.3)
        s1=scatter(1:size(projectedX,2),projectedX(i,:),5,'filled','MarkerEdgeColor','none','MarkerFaceAlpha',.2,'MarkerFaceColor',p(k).Color);
        uistack(s1,'bottom')
        %plot(model{k}.Xp(i,:),'LineWidth',1,'DisplayName',nn,'Color','k');
        if k==length(model)
            legend(findobj(gca,'Type','Line','LineWidth',2),'Location','Best')
            title('(Smoothed) States vs. projection')
            ylabel(['State ' num2str(i)])
        end
        axis tight
        grid on

        %subplot(Nx,Ny,Ny*(i-1)+4) %State residuals
        %hold on
        %set(gca,'ColorOrderIndex',k)
        %scatter(1:size(projectedX,2),Xs(i,:)-projectedX(i,:),5,'filled','MarkerEdgeColor','none','MarkerFaceAlpha',.5);
        %if k==length(model)
        %    title('(KS) State residual (vs. projection)')
        %    grid on
        %end
        %axis tight
        
        subplot(Nx,Ny,Ny*(i-1)+4) %State residuals
        hold on
        %Smooth states
        set(gca,'ColorOrderIndex',k)
        %Det states:
        p(k,i)=plot(Xsim(i,:),'LineWidth',2,'DisplayName',nn);
        patch([1:size(Xsim,2),size(Xsim,2):-1:1]',[Xsim(i,:)+sqrt(squeeze(Psim(i,i,:)))', fliplr(Xsim(i,:)-sqrt(squeeze(Psim(i,i,:)))')]',p(k).Color,'EdgeColor','none','FaceAlpha',.3)
        s1=scatter(1:size(projectedX,2),projectedX(i,:),5,'filled','MarkerEdgeColor','none','MarkerFaceAlpha',.2,'MarkerFaceColor',p(k).Color);
        uistack(s1,'bottom')
        if k==length(model)
            legend(findobj(gca,'Type','Line','LineWidth',2),'Location','Best')
            title('(Deterministic) States vs. projection')
            ylabel(['State ' num2str(i)])
        end
        axis tight
        grid on
    end
end

%% MLE state innovation
subplot(Nx,Ny,Ny*(M)+4)
hold on
for k=1:length(model)
    stError=dFit{k}.MLEstate.state-dFit{k}.oneAheadMLE.state(:,1:end-1);
    [~,z2]=logLnormal(stError,model{k}.Q); %This should be the innovation z2 score, but that would require to also consider the prior uncertainty, which requires a for-loop here. 
    p1=plot(aux1,'LineWidth',1);
    bar2=bar([yoff + k*100],nanmean([aux1]),'EdgeColor','none','BarWidth',100,'FaceColor',p1.Color);
    text(yoff+(k)*100,nanmean(aux1)*(1+.3*k),[num2str(nanmean(aux1))],'Color',bar2.FaceColor)
end
title('(KS) Predicted state error (z-score)')
axis tight
grid on

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
       [~,modelOut]=datSet.getOneAheadResiduals(dFit{k});
       dd=trueD-modelOut(:,viewPoints(i)+[-(binw/2):(binw/2)]);
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
