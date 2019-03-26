function [fh] = vizDataLikelihood(model,datSet)
if ~iscell(datSet)
    datSet={datSet};
end
if ~isa(model{1},'struct')
  %To do: convert struct to model
end
M=max(cellfun(@(x) size(x.A,1),model(:)));
fh=figure('Units','Normalized','OuterPosition',[.25 .4 .5 .3]);

%LogL, BIC, AIC
mdl=model;
Md=length(datSet);
for kd=1:Md %One row of subplots per dataset
    dFit=cellfun(@(x) x.fit(datSet{kd}),model,'UniformOutput',false);
    logLtest=cellfun(@(x) x.logL,dFit);
    yy=logLtest;
    nn='logL';
    subplot(Md,1,kd)
    hold on
    Mm=length(mdl);
    DeltaIC=yy-max(yy);
    modelL=exp(DeltaIC);
    w=modelL/sum(modelL); %Computes bayes factors, or Akaike weights. See Wagenmakers and Farrell 2004, Akaike 1978, Kass and Raferty 1995
    for k=1:Mm
        set(gca,'ColorOrderIndex',k)
        bar2=bar([k*100],yy(k),'EdgeColor','none','BarWidth',100);
        text((k)*100,.982*(yy(k)),[num2str(yy(k),6)],'Color','w','FontSize',8,'Rotation',90)
    end
    set(gca,'XTick',[1:Mm]*100,'XTickLabel',cellfun(@(x) x.name, mdl,'UniformOutput',false),'XTickLabelRotation',90)
    title([nn])
    grid on
    axis tight;
    aa=axis;
    axis([aa(1:2) .98*min(yy) max(yy)])
end
end
