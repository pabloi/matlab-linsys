function [fh,fh2] = vizDataRes(model,Y,U)

M=max(cellfun(@(x) size(x.J,1),model(:)));
yoff=size(Y,2)*1.1;
Nm=length(model); %Number of models
%% Add reduced models:
for i=1:length(model)
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
        [model{i}.J,model{i}.B,model{i}.C,~,~,model{i}.Q] = canonize(model{i}.J,model{i}.B,model{i}.C,[],model{i}.Q,[],'canonicalAlt');
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

%% Plot features of one-ahead output residuals:
fh2=figure('Units','Normalized','OuterPosition',[0 0 1 1]);
Nny=5;
Nx=8;
allLogL=cellfun(@(x) x.logLtest,model);
bestModel=find(allLogL==min(allLogL));
res=Y(:,2:end)-model{bestModel}.oneAheadOutF; %One-ahead residuals
res=substituteNaNs(res'); %Removing NaNs, otherwise this is all crap
[pp,cc,aa]=pca((res),'Centered','off');
for i=1:length(model)
    res=Y(:,2:end)-model{i}.oneAheadOutF; %One-ahead residuals
    cc=(pp\res)';

    for kk=1:Nx
        subplot(Nx,Nny,(kk-1)*Nny+1)
        if model{i}.logLtest==min(allLogL)
          Nc=size(pp,1);
            try
                imagesc(flipud(reshape(pp(:,kk),12,Nc/12)'))
            catch
                imagesc(flipud(pp(:,kk)))
            end
            colormap(flipud(map))
            aC=.5*max(abs(pp(:)));
            caxis([-aC aC])
            title(['Residual PC ' num2str(kk) ' (of best model)'])
        end

        subplot(Nx,Nny,(kk-1)*Nny+2)
        hold on
        %binw=5;
        %aux1=conv(cc(:,kk)',ones(1,binw)/binw,'valid');
        %p(i)=plot(aux1,'LineWidth',1) ;
        p(i)=scatter(1:length(cc),cc(:,kk),5,'filled','MarkerEdgeColor','none','MarkerFaceAlpha',.2);
        title('PC of residual, mov. avg.')
        grid on

        subplot(Nx,Nny,(kk-1)*Nny+3)
        hold on
        qq1=qqplot(cc(:,kk));
        qq1(1).MarkerEdgeColor=p(i).CData;
        ax=gca;
        ax.Title.String=['QQ plot residual PC ' num2str(kk)];

        subplot(Nx,Nny,(kk-1)*Nny+4)
        hold on
        r=xcorr(cc(:,kk));
        plot(-(length(r)-1)/2:(length(r)-1)/2,r)
        axis tight
        aa=axis;
        grid on
        xlabel('Delay (samp)')
        title('Residual PC 1 autocorr')
        axis([-15 15 aa(3:4)])

        subplot(Nx,Nny,(kk-1)*Nny+5)
        hold on
        histogram(cc(:,kk),'EdgeColor','none','Normalization','pdf','FaceAlpha',.2,'BinEdges',[-1:.02:1])
        title(['Residual PC ' num2str(kk) ' histogram'])
    end
end
