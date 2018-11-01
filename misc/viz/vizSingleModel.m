function [fh] = vizSingleModel(singleModel,Y,U)

M=size(singleModel.J,1);
fh=figure('Units','Normalized','OuterPosition',[0 0 .5 1],'Color',ones(1,3));
if nargin>1
    Nx=10;
else
    Nx=4;
end
Nu=size(singleModel.B,2);
Ny=M+Nu+1;
model{1}=singleModel;
Nc=size(Y,1);
N=size(Y,2);

%% First: normalize model, compute basic parameters of interest:
if nargin<3
    U=[zeros(Nu,100) ones(Nu,1000)]; %Step response
end
for i=1:length(model)
    [model{i}.J,model{i}.B,model{i}.C,~,~,model{i}.Q] = canonize(model{i}.J,model{i}.B,model{i}.C,[],model{i}.Q,[],'canonical',900);
    [Y2,X2]=fwdSim(U,model{i}.J,model{i}.B,model{i}.C,model{i}.D,[],[],[]); %Simulating from MLE initial state
    model{i}.smoothStates=X2;
    model{i}.smoothOut=Y2;
    model{i}.logLtest=dataLogLikelihood(Y,U,model{i}.J,model{i}.B,model{i}.C,model{i}.D,model{i}.Q,model{i}.R,[],[],'approx');
    [bic,aic,bic2]= bicaic(model{i},Y,numel(Y)*model{i}.logLtest);
    model{i}.BIC=bic/(2*numel(Y)); %To put in the same scale as logL
    model{i}.BIC2=bic2/(2*numel(Y)); %To put in the same scale as logL
    model{i}.AIC=aic/(2*numel(Y));
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
set(fh,'Name',['Per sample logL=' num2str(model{1}.logLtest) ', BIC=' num2str(model{1}.BIC) ', AIC=' num2str(model{1}.AIC) ',BICalt=' num2str(model{1}.BIC2)]);

%% Define colormap:
ex1=[1,0,0];
ex2=[0,0,1];
mid=ones(1,3);
N=100;
gamma=1;
map=[flipud(mid+ (ex1-mid).*([1:N]'/N).^gamma); mid; (mid+ (ex2-mid).*([1:N]'/N).^gamma)];

%% Plo
ytl={'GLU','TFL','ADM','HIP','RF','VL','VM','SMT','SMB','BF','MG','LG','SOL','PER','TA'};
yt=1:15;
fs=7;
% STATES
C=model{1}.C;
D=model{1}.D;
CD=[C D];
%Rotate factors for display:
XU=[model{1}.Xs;U];
rotMed='orthonormal';
rotMed='orthomax';
%rotMed='promax';
%rotMed='quartimax';
%rotMed='pablo';
%rotMed='none';
[CDrot,XUrot]=rotateFac(CD,XU,rotMed);
if strcmp(rotMed,'none')
    factorName=[strcat('C_',num2str([1:size(model{1}.C,2)]'));strcat('D_',num2str([1:size(model{1}.D,2)]'))];
    latentName=[strcat('State ',' ', num2str([1:size(model{1}.C,2)]'));strcat('Input ',' ', num2str([1:size(model{1}.D,2)]'))];
else
    factorName=strcat('Factor ',num2str([1:size(CD,2)]'));
    latentName=strcat('Latent ',num2str([1:size(CD,2)]'));
end
%
aC=prctile(abs(Y(:)),98);
CiR=CDrot'*inv(model{1}.R);
CiRC=CiR*CDrot;
projY=CiRC\CiR*Y; %Minimum variance projection, kalman-like
Nd=size(model{1}.J,1);
clear ph
for k=1:size(CDrot,2)
    ph(k)=subplot(Nx,Ny,k); %passthrough term
    hold on
    if nargin>1 %Adding data projection onto dim
        scatter(1:size(Y,2),projY(k,:),5,.7*ones(1,3),'filled')
    end
    plot(XUrot(k,:),'k','LineWidth',2);
    title(latentName(k,:))
    ax=gca;
    ax.Position=ax.Position+[0 .02 0 0];
    grid on
    axis tight

    subplot(Nx,Ny,Ny+k+[0,Ny])
    try
        imagesc((reshape(CDrot(:,k),12,Nc/12)'))
        set(gca,'XTick',[],'YTick',yt,'YTickLabel',ytl,'FontSize',fs)
    catch
        imagesc((CD(:,k)))
        set(gca,'XTick',[],'YTick',[],'YTickLabel',[],'FontSize',fs)
    end

    colormap(flipud(map))
    %aC=max(abs(CD(:)));
    %aC=prctile(abs(Y(:)),99);
    caxis([-aC aC])
    axis tight
    if k==1
        ylabel({'Contribution';'to output'})
    end
    ax=gca;
    ax.YAxis.Label.FontSize=12;
    ax.YAxis.Label.FontWeight='bold';
    title(factorName(k,:))
end
linkaxes(ph,'y');
%Covariances
if strcmp(rotMed,'none') %If things are rotated, this matrix cannot be associated to specific states
    subplot(Nx,Ny,Ny)
    imagesc(model{1}.Q)
    set(gca,'XTick',[],'YTick',[],'YTickLabel',[],'FontSize',8)
    colormap(flipud(map))
    aQ=.5*max(abs(model{1}.Q(:)));
    caxis([-aQ aQ])
    axis tight
    title(['Q, tr(Q)=' num2str(trace(model{1}.Q))])
end
subplot(Nx,Ny,2*Ny+[0,Ny])
imagesc(model{1}.R)
set(gca,'XTick',[],'YTick',[],'YTickLabel',[],'FontSize',8)
colormap(flipud(map))
aR=.5*max(abs(model{1}.R(:)));
caxis([-aR aR])
axis tight
title(['R, tr(R)=' num2str(trace(model{1}.R))])

if nargin<2
    %Third row: one-ahead step-response


else %IF DATA PRESENT:
N=size(Y,2);
viewPoints=[1,140,151,251,651,1040,1051,1151,1301,N-11]+5;
%viewPoints=[151,1034,1051]+8;
binw=4; %plus minus 2 strides
viewPoints(viewPoints>N-binw/2)=[];
Ny=length(viewPoints);
M=length(model);
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
        end

        subplot(Nx,Ny,i+(1+2*k)*Ny+[0,Ny])
        try
            imagesc(reshape(nanmean(dd,2),12,size(Y,1)/12)')
            set(gca,'XTick',[],'YTick',yt,'YTickLabel',ytl,'FontSize',fs)
        catch
            imagesc(nanmean(dd,2))
            set(gca,'XTick',[],'YTick',[],'YTickLabel',[],'FontSize',fs)
        end
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
Ny=2;
subplot(Nx,Ny,1+9*Ny)
hold on
dd=model{1}.oneAheadRes;
aux1=sqrt(sum(dd.^2));
binw=10;
aux1=conv(aux1,ones(1,binw)/binw,'valid'); %Smoothing
p1=plot(aux1,'LineWidth',1,'DisplayName','Residual RMSE, 10-stride mov. avg.');
dd=Y-C*(C\(Y-D*U))-D*U;
aux1=sqrt(sum(dd.^2));
binw=10;
aux1=conv(aux1,ones(1,binw)/binw,'valid'); %Smoothing
p1=plot(aux1,'LineWidth',1,'DisplayName','Output orthogonal to [C,D], RMSE, 10-stride mov. avg.');
%title('MLE one-ahead output error (RMSE, mov. avg.)')
axis tight
grid on
set(gca,'YScale','log')
ind=find(diff(U(1,:))~=0); %Bad way to do this
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

subplot(Nx,Ny,2+9*Ny)
hold on
title('Autocorr of residuals first 3 PCs')
dd=model{1}.oneAheadRes;
dd=substituteNaNs(dd');
[pp,cc,aa]=pca((dd),'Centered','off');
for k=1:3
r=xcorr(cc(:,k));
plot(-(length(r)-1)/2:(length(r)-1)/2,r)
end
axis tight
aa=axis;
grid on
xlabel('Delay (samp)')
title('Residual PC 1 autocorr')
axis([-15 15 aa(3:4)])

%hold on
%aux1=conv(cc(:,1)',ones(1,binw)/binw,'valid');
%plot(aux1,'LineWidth',1) ;
%title('First PC of residual, mov. avg.')
%grid on



end
