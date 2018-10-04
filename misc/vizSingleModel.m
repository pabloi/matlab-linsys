function [fh] = vizSingleModel(singleModel,Y,U)

M=size(singleModel.J,1);
fh=figure('Units','Normalized','OuterPosition',[0 0 .5 1],'Color',ones(1,3));
if nargin>1
    Nx=10;
else
    Nx=4;
end
Ny=M+2;
model{1}=singleModel;
Nu=size(model{1}.B,2);
Nc=size(Y,1);

%% First: normalize model, compute basic parameters of interest:
if nargin<3
    U=[zeros(Nu,100) ones(Nu,1000)]; %Step response
end
for i=1:length(model)
    [model{i}.J,model{i}.B,model{i}.C,~,~,model{i}.Q] = canonizev5(900,model{i}.J,model{i}.B,model{i}.C,[],model{i}.Q);
    [Y2,X2]=fwdSim(U,model{i}.J,model{i}.B,model{i}.C,model{i}.D,[],[],[]); %Simulating from MLE initial state
    model{i}.smoothStates=X2;
    model{i}.smoothOut=Y2;
    model{i}.logLtest=dataLogLikelihood(Y,U,model{i}.J,model{i}.B,model{i}.C,model{i}.D,model{i}.Q,model{i}.R,[],[],'approx');
    if nargin>1
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
        model{i}.oneAheadRes=Y(:,2:end)-model{i}.oneAheadOut;

    end
end

%% Define colormap:
ex1=[1,0,0];
ex2=[0,0,1];
mid=ones(1,3);
N=100;
map=[ex1.*[N:-1:1]'/N + mid.*[0:N-1]'/N; mid; ex2.*[0:N-1]'/N + mid.*[N:-1:1]'/N];

%% Plo
ytl={'GLU','TFL','ADM','HIP','RF','VL','VM','SMT','SMB','BF','MG','LG','SOL','PER','TA'};
yt=1:15;
fs=7;
% STATES
CD=[model{i}.C model{i}.D];
%CDiR=CD'*inv(model{i}.R);
%CDiRCD=CDiR*CD;
%projY=CDiRCD\CDiR*Y;
projY=CD\Y;
subplot(Nx,Ny,1) %passthrough term
hold on
if nargin>1
    scatter(1:size(Y,2),projY(M+1,:),5,.7*ones(1,3),'filled')
end
p(i)=plot(U,'k','LineWidth',2,'DisplayName',['Passthrough term, \tau=0']);
title('Input')
ax=gca;
ax.Position=ax.Position+[0 .02 0 0];
subplot(Nx,Ny,Ny+1+[0,Ny])
try
    imagesc((reshape(model{1}.D,12,Nc/12)'))
    set(gca,'XTick',[],'YTick',yt,'YTickLabel',ytl,'FontSize',fs)
catch
    imagesc((model{1}.D))
    set(gca,'XTick',[],'YTick',[],'YTickLabel',[],'FontSize',fs)
end

colormap(flipud(map))
aC=max(abs(CD(:)));
caxis([-aC aC])
axis tight
ylabel({'Contribution';'to output'})
ax=gca;
ax.YAxis.Label.FontSize=12;
ax.YAxis.Label.FontWeight='bold';
title('D')

for i=1:M
    subplot(Nx,Ny,i+1) %TOP row: states temporal evolution and data projection
    hold on
    if nargin>1
        scatter(1:size(Y,2),projY(i,:),5,.7*ones(1,3),'filled')
    end
    set(gca,'ColorOrderIndex',1)
    %p(i)=plot(model{1}.smoothStates(i,:),'LineWidth',2,'DisplayName',['Deterministic state, \tau=' num2str(-1./log(model{1}.J(i,i)),3)]);
    title(['State ' num2str(i)])
    %title('(Smoothed) Step-response states')
    p(i)=plot(model{1}.Xf(i,:),'LineWidth',2,'DisplayName',['\tau=' num2str(-1./log(model{1}.J(i,i)),3)],'Color','k');
    patch([1:size(model{1}.Xf,2),size(model{1}.Xf,2):-1:1]',[model{1}.Xf(i,:)+sqrt(squeeze(model{1}.Pf(i,i,:)))', fliplr(model{1}.Xf(i,:)-sqrt(squeeze(model{1}.Pf(i,i,:)))')]',p(i).Color,'EdgeColor','none','FaceAlpha',.3)
    legend(p(i),'Location','SouthWest')
    ax=gca;
    ax.Position=ax.Position+[0 .02 0 0];

    subplot(Nx,Ny,Ny+i+1+[0,Ny])% Second row: checkerboards
    try
        imagesc((reshape(model{1}.C(:,i),12,Nc/12)'))
        set(gca,'XTick',[],'YTick',yt,'YTickLabel',[],'FontSize',fs)
    catch
        imagesc((model{1}.C(:,i)))
        set(gca,'XTick',[],'YTick',[],'YTickLabel',[],'FontSize',fs)
    end

    ax=gca;
    ax.YAxis.Label.FontSize=12;
    colormap(flipud(map))
    caxis([-aC aC])
    axis tight
    title(['C_' num2str(i)])
end

%Covariances
subplot(Nx,Ny,Ny)
imagesc(model{1}.Q)
set(gca,'XTick',[],'YTick',[],'YTickLabel',[],'FontSize',8)
colormap(flipud(map))
aC=.5*max(abs(model{1}.Q(:)));
caxis([-aC aC])
axis tight
subplot(Nx,Ny,2*Ny+[0,Ny])
imagesc(model{1}.R)
set(gca,'XTick',[],'YTick',[],'YTickLabel',[],'FontSize',8)
colormap(flipud(map))
aC=.5*max(abs(model{1}.R(:)));
caxis([-aC aC])
axis tight

if nargin<2
    %Third row: one-ahead step-response


else %IF DATA PRESENT:
N=size(Y,2);
viewPoints=[1,40,51,151,251,651,940,951,1001,1101,N-11]+5;
%viewPoints=[51,934,951]+8;
binw=4;
viewPoints(viewPoints>N-binw/2)=[];
Ny=length(viewPoints);
M=length(model);
aC=max(abs(model{1}.C(:)));
for k=1:3
    for i=1:Ny
        switch k
        case 1 % Third row, actual data
            dd=Y(:,viewPoints(i)+[-(binw/2):(binw/2)]);
            nn='Data';
        case 2 %Fourth row: one-ahead data predictions
            dd=model{1}.oneAheadOut(:,viewPoints(i)+[-(binw/2):(binw/2)]);
            nn={'MLE Prediction';'(one-step ahead)'};
        case 3 % Fifth row:  data residuals (checkerboards)
            dd=model{1}.oneAheadRes(:,viewPoints(i)+[-(binw/2):(binw/2)]);
            nn='Residual';
            aC=max(abs(model{1}.C(:)));; %2x magnification of residuals
        end

        subplot(Nx,Ny,i+(1+2*k)*Ny+[0,Ny])
        try
            imagesc(reshape(nanmean(dd,2),12,size(Y,1)/12)')
            set(gca,'XTick',[],'YTick',yt,'YTickLabel',ytl,'FontSize',fs)
        catch
            imagesc(nanmean(dd,2))
            set(gca,'XTick',[],'YTick',[],'YTickLabel',[],'FontSize',fs)
        end
        %if i==1

    %else
%set(gca,'XTick',[],'YTick',yt,'YTickLabel',[],'FontSize',fs)
%    end
        ax=gca;

        colormap(flipud(map))
        caxis([-aC aC])
        axis tight
        if k==1
            title(['Output at t=' num2str(viewPoints(i))])
            ax=gca;
            ax.Title.FontSize=10;
        end
        if i==1
            ylabel(nn)
            ax=gca;
            ax.YAxis.Label.FontWeight='bold';
            ax.YAxis.Label.FontSize=12;
        end
    end
end

% Sixth row: residual RMSE, Smoothed, first PC of residual, variance by itself
Ny=1;
subplot(Nx,Ny,1+9*Ny)
hold on
dd=model{1}.oneAheadRes;
%dd=Y-CD*projY;
aux1=sqrt(sum(dd.^2));
binw=10;
aux1=conv(aux1,ones(1,binw)/binw,'valid'); %Smoothing
p1=plot(aux1,'LineWidth',1,'DisplayName','Residual RMSE, 10-stride mov. avg.');
dd=Y-CD*projY;
aux1=sqrt(sum(dd.^2));
binw=10;
aux1=conv(aux1,ones(1,binw)/binw,'valid'); %Smoothing
p1=plot(aux1,'LineWidth',1,'DisplayName','Output orthogonal to [C,D], RMSE, 10-stride mov. avg.');
%title('MLE one-ahead output error (RMSE, mov. avg.)')
axis tight
grid on
set(gca,'YScale','log')
ind=find(diff(U)~=0);
Y(:,ind)=nan;
aux1=conv2(Y,[-.5,1,-.5]/sqrt(1.5),'valid'); %Y(k)-.5*(y(k+1)+y(k-1));
%aux1=(Y(:,2:end)-Y(:,1:end-1))/sqrt(2);
aux1=sqrt(sum(aux1.^2));
aux1=conv(aux1,ones(1,binw)/binw,'valid'); %Smoothing
plot(aux1,'LineWidth',1,'DisplayName','''Instantaneous std''') ;
ylabel({'Residual';' RMSE'})
ax=gca;
ax.YAxis.Label.FontSize=12;
ax.YAxis.Label.FontWeight='bold';
legend('Location','NorthEast')

%subplot(Nx,Ny,2+9*Ny)
%[pp,cc,aa]=pca((dd'),'Centered','off');
%hold on
%aux1=conv(cc(:,1)',ones(1,binw)/binw,'valid');
%plot(aux1,'LineWidth',1) ;
%title('First PC of residual, mov. avg.')
%grid on

%subplot(Nx,Ny,3+9*Ny)
%hold on
%aux1=conv2(Y,[-.5,1,-.5],'valid'); %Y(k)-.5*(y(k+1)+y(k-1));
%aux1=sqrt(sum(aux1.^2))/sqrt(1.5);
%aux1=aux1./(8.23+sqrt(sum(Y(:,2:end-1).^2))); %
%aux1=conv(aux1,ones(1,binw)/binw,'valid'); %Smoothing
%plot(aux1,'LineWidth',1) ;
%title('Instantaneous normalized std of data')
%grid on
%set(gca,'YScale','log')

end
