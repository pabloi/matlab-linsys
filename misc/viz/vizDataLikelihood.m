function [fh,fh2] = vizDataLikelihood(model,datSet)
if ~isa(model{1},'struct')
  %To do: convert struct to model
end
M=max(cellfun(@(x) size(x.A,1),model(:)));
fh=figure('Units','Normalized','OuterPosition',[.25 .4 .5 .3]);
Ny=4;
Nx=max(M,6);
yoff=size(datSet.out,2)*1.1;
Nm=length(model); %Number of models
dFit=cellfun(@(x) x.fit(datSet),model,'UniformOutput',false);
logLtest=2*cellfun(@(x) x.logL,dFit);
BIC=-cellfun(@(x) x.BIC,dFit);
AIC=-cellfun(@(x) x.AIC,dFit);

%LogL, BIC, AIC
mdl=model;
for kj=1:3 %logL, BIC, aic
    switch kj
        case 1
            yy=logLtest;
            nn='2 logL';
        case 3
            yy=BIC;
            nn='-BIC';
        case 2
            yy=AIC;
            nn='-AIC';
    end
    subplot(1,3,kj)
    hold on
    [N,Nz]=size(datSet.out);
    Mm=length(mdl);
    for k=1:Mm
        set(gca,'ColorOrderIndex',k)
        bar2=bar([k*100],yy(k),'EdgeColor','none','BarWidth',100);
        text((k)*100,.982*(yy(k)),[num2str(yy(k),6)],'Color','w','FontSize',8,'Rotation',90)
        if kj==1 && k>1
          deltaDof=mdl{k}.dof-mdl{k-1}.dof;
          text((k)*100-50,1*(yy(k)),['dof=' num2str(deltaDof) ],'Color','k','FontSize',6)
          text((k)*100-50,.9975*(yy(k)),['\chi^2=' num2str(yy(k)-yy(k-1),4)],'Color','k','FontSize',6)
          text((k)*100-50,.995*(yy(k)),['p=' num2str(chi2inv(yy(k)-yy(k-1),deltaDof),3)],'Color','k','FontSize',6)
        end
    end
    set(gca,'XTick',[1:Mm]*100,'XTickLabel',cellfun(@(x) x.name, mdl,'UniformOutput',false),'XTickLabelRotation',90)
    title([nn])
    grid on
    %set(gca,'XTick',100*(Mm+1)*.5,'XTickLabel',nn)
    axis tight;
    aa=axis;
    axis([aa(1:2) .98*min(yy) max(yy)])
end
