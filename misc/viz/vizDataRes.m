function [fh2] = vizDataRes(model,Y,U)
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
yoff=size(Y,2)*1.1;
Nm=length(model); %Number of models
%% Compute fits
for i=1:length(model)
    dFit{i}=model{i}.fit(datSet);
end

%% Define colormap:
ex1=[1,0,0];
ex2=[0,0,1];
mid=ones(1,3);
N=100;
map=[ex1.*[N:-1:1]'/N + mid.*[0:N-1]'/N; mid; ex2.*[0:N-1]'/N + mid.*[N:-1:1]'/N];

%% Plot features of one-ahead output residuals:
fh2=figure('Units','Normalized','OuterPosition',[0 0 1 1]);
Nny=5;
Nx=6;%8;
allLogL=cellfun(@(x) x.logL,dFit);
bestModel=find(allLogL==min(allLogL),1,'first');
res=datSet.getOneAheadResiduals(dFit{bestModel});
res=substituteNaNs(res'); %Removing NaNs, otherwise this is all crap
[pp,cc,aa]=pca((res),'Centered','off');
for i=1:length(model)
    res=datSet.getOneAheadResiduals(dFit{i});
    cc=(pp\res)';

    for kk=1:Nx
        subplot(Nx,Nny,(kk-1)*Nny+1)
        if dFit{i}.logL==min(allLogL)
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
