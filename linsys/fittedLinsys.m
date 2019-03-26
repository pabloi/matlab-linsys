classdef fittedLinsys < linsys
%This class extends linsys to include additional fields/methods relevant for models that are fitted through some optimization.
%It facilitates model selection by storing data of parameters fitted, logL achieved, etc.

properties
  goodnessOfFit %log-L for EM, could be RMSE for LS, etc.
  initCondPrior=initCond([],[]);
  trainOptions=struct();
  method='';
  dataSetHash=''; %Just the hash, to avoid storing all the data
  trainingLog=[]; %Log of training
  dataSetInput %This is stored to compute the fitted output values
  dataSetNonNaNSamples
end
properties (Dependent)
  BIC
  AIC
  AICc
  fittedOutput
  fittedResiduals
end
methods
  function this=fittedLinsys(A,C,R,B,D,Q,iC,dataSet,methodName,opts,gof,outlog)
      this@linsys(A,C,R,B,D,Q);
      if nargin>9
      this.trainOptions=opts;
      end
      if nargin>8
      this.method=methodName;
      end
      if nargin>7
      this.dataSetHash=dataSet.hash;
      this.dataSetInput=dataSet.in;
      this.dataSetNonNaNSamples=dataSet.nonNaNSamp;
      end
      if nargin>10
      this.goodnessOfFit=gof;
      end
      if nargin>6 && ~isempty(iC)
        this.initCondPrior=iC; %Should be initCond object
      end
      if nargin>11
      this.trainingLog=outlog;
      end
  end
  function df=dof(this)
      %Computes effective degrees of freedom of the system, assuming all non-zero parameters were freely selected.
      %Warning: this presumes a diagonal A matrix, and counting non-zero entries post-diagonalization. This is not necessarily the way that results in the least amount of non-zero parameters (can this be proved?).
      Nu=this.Ninput;
      Nx=this.order;
      Ny=this.Noutput;
      opts=this.trainOptions;
      if strcmp(this.method,'repeatedEM') || strcmp(this.method,'EM')
          opts=processEMopts(opts,Nu,Nx,Ny);
      else
          error('The fitting method is unknown, cannot count free parameters');
      end
      % Degrees of freedom of A
      if ~isempty(opts.fixA)
        Na=0;
      else
        Na=Nx; %A has only the eigenvalues as true dofs
      end
      %B dof
      if ~isempty(opts.fixB)
          Nb=0;
        else
          Nb=(length(opts.indB))*Nx;
      end
      %C dof
      if ~isempty(opts.fixC)
          Nc=0;
        else
          Nc=length(opts.includeOutputIdx)*Nx; %includeOutputIdx prevents some outputs from influencing the dynamics equation, equivalent to having infinte variance in the corresponding R entries
      end
      %D dof
      if ~isempty(opts.fixD)
          Nd=0;
        else
          Nd=(length(opts.indD))*Ny; %includeOutputIdx does NOT exclude any outputs from the computation of D
      end
      %Q dof
      if ~isempty(opts.fixQ)
          Nq=0;
        else
          Nq=Nx*(Nx+1)/2;
      end
      %Rdof:
      Nr=this.Rdof; %This is its own function because it is needed for AICc

      Nbcq= Nb+Nc+Nq-Nx;  %Up to Nx parameters can be fixed in B, C, or Q, without losing degrees of freedom (you just lose redundant representations).
      %For example, the first column of B can be arbitrarily set to all 1,
      %or the norm of columns of C to be all 1 too, or the diagonal
      %elements of Q be set to 1. These are essentially scale parameters.
      df=Na+Nbcq+Nd+Nr; %Model free parameters
      if this.order==1 && this.A==0 %Flat model gets expressed as a 1st order
         df=Nd+Nr;
      end
  end
  function df=Rdof(this)
    %Compute the degrees of freedom of R alone
    Nu=this.Ninput;
    Nx=this.order;
    Ny=this.Noutput;
    opts=this.trainOptions;
    if strcmp(this.method,'repeatedEM') || strcmp(this.method,'EM')
        opts=processEMopts(opts,Nu,Nx,Ny);
    else
        error('The fitting method is unknown, cannot count free parameters');
    end
    if ~isempty(opts.fixR)
      Nr=0;
    elseif opts.diagR %Diagonal estimation
      Nr=length(opts.includeOutputIdx);
    else %Not fixed, non diag R
      M=length(opts.includeOutputIdx);
      Nr=M*(M+1)/2;
    end
    df=Nr;
  end
  function bic=get.BIC(this)
    if strcmp(this.method,'EM') || strcmp(this.method,'repeatedEM')
      logL=this.goodnessOfFit;
      bic=-2*logL+log(this.dataSetNonNaNSamples)*this.dof;
    else
      error('BIC is not defined unless goodness of fit metric is logL')
    end
  end
  function aic=get.AIC(this)
    if strcmp(this.method,'EM') || strcmp(this.method,'repeatedEM')
      logL=this.goodnessOfFit;
      aic=-2*logL+2*this.dof;
    else
      error('AIC is not defined unless goodness of fit metric is logL')
    end
  end
  function aicc=get.AICc(this)
      if strcmp(this.method,'EM') || strcmp(this.method,'repeatedEM')
        p=this.dof;
        N=this.dataSetNonNaNSamples;
        v=this.Rdof;
        aicc=this.AIC+2*p*(p+v)/(N*this.Noutput-p-v);
        %This expression is drawn from Burnham and Anderson 2002, eq. 7.91
      else
        error('AICc is not defined unless goodness of fit metric is logL')
      end
  end
  function [p,chi,deltaDof]=likelihoodRatioTest(this,other)
    %Performs a likelihood ratio test for the fits of two models to a given dataset.
    %Uses Wilk's theorem approximation for the asymptotic distribution of
    %the likelihood ratio.
    %CAUTION: this is known to NOT be a valid approximation for LTI-SSM
    %modeling of different orders.
    %p returns the p-value of the observed likelihood ratio (of the model with more parameters over the one with less, so the ratio should always be larger than 1)
    %Models should be nested for this to work.
    %Check: both dataFits refer to the same dataset
    if this.dataSetHash ~= other.dataSetHash
      error('dataFit:LRT','Performing likelihood ratio test on fits of different datasets')
    end
    if ~(strcmp(this.method,'EM') || strcmp(this.method,'repeatedEM')) || ~(strcmp(other.method,'EM') || strcmp(other.method,'repeatedEM'))
      error('LRT is not defined unless goodness of fit metric is logL')
    end
    deltaDof=this.dof-other.dof;
    if deltaDof<0
      deltaDof=-deltaDof;
    end
    chi=2*(this.goodnessOfFit - other.goodnessOfFit);
    if chi<2
      warning('dataFit:LRT','Model with more parameters has lower likelihood. This means either a bad fit or that models were not nested.')
    end
    p=1-chi2cdf(chi,deltaDof);
  end
end
methods(Static)
    function fh=compare(fittedModels)
        %Shows LRT, AIC, AICc, BIC for a collection of fitted models to the
        %same data.
        if ~isa(fittedModels,'cell') && ~isa(fittedModels{1},'fittedLinsys')
            error('Input argument must be a cell array containing fittedLinsys objects')
        end
        %To do: check that ALL models were fitted to the same dataset with the same method,
        %otherwise the comparison is meaningless

        %Create a figure with the relevant plots:
        M=max(cellfun(@(x) size(x.A,1),fittedModels(:)));
        fh=figure('Units','Normalized','OuterPosition',[.25 .4 .5 .3]);

        %LogL, BIC, AIC, AICc
            logLtest=cellfun(@(x) x.goodnessOfFit,fittedModels);
            BIC=-cellfun(@(x) x.BIC,fittedModels)/2;
            AIC=-cellfun(@(x) x.AIC,fittedModels)/2;
            AICc=-cellfun(@(x) x.AICc,fittedModels)/2;
            for kj=1:4 %logL, BIC, aic, aicc: constrain BIC,AIC, AICc and LRT stats on logL for fittedLinsys that were fitted to this particular dataset
                switch kj
                    case 1
                        yy=logLtest;
                        nn='\Delta logL';
                    case 2
                        yy=BIC;
                        nn='-\Delta BIC/2';
                    case 3 %Never used
                        yy=AIC;
                        nn='-\Delta AIC/2';
                    case 4
                        yy=AICc;
                        nn='-\Delta AICc/2';
                end
                subplot(1,4,kj)
                hold on
                Mm=length(fittedModels);
                DeltaIC=yy-max(yy);
                modelL=exp(DeltaIC);
                w=modelL/sum(modelL); %Computes bayes factors, or Akaike weights. See Wagenmakers and Farrell 2004, Akaike 1978, Kass and Raferty 1995
                yy=yy-min(yy);
                My=max(max(yy),0);
                my=min(min(yy),0);
                for k=1:Mm
                    set(gca,'ColorOrderIndex',k)
                    bar2=bar([k*100],yy(k),'EdgeColor','none','BarWidth',100);
                    text((k)*100,.01*yy(k),[num2str(yy(k),6)],'Color','w','FontSize',8,'Rotation',90)
                    if kj==1 && k>1
                      [pval,chi,deltaDof]=likelihoodRatioTest(fittedModels{k},fittedModels{k-1});
                      text((k)*100-40,.75*yy(k),['\chi^2_{' num2str(deltaDof) '}=' num2str(round(chi))],'Color','w','FontSize',6)
                      if pval>1e-9
                         text((k)*100-30,.9*yy(k),['p=' num2str(pval,2)],'Color','w','FontSize',6)
                      else
                         text((k)*100-30,.9*yy(k),['p<1e-9'],'Color','w','FontSize',6)
                      end
                    elseif kj ~=1
                      text((k)*100-35,.9*yy(k),[ num2str(round(100*w(k))) '%'],'Color','w','FontSize',8)
                    end
                end
                set(gca,'XTick',[1:Mm]*100,'XTickLabel',cellfun(@(x) x.name, fittedModels,'UniformOutput',false),'XTickLabelRotation',90)
                title([nn])
                grid on
                %set(gca,'XTick',100*(Mm+1)*.5,'XTickLabel',nn)
                axis tight;
                aa=axis;
                deltaYY=My-my;
                axis([aa(1:2) my-.01*deltaYY 1.1*My])
            end
    end
end
end
