classdef dataFit

properties (SetAccess = immutable)
  model %linsys object (model)
  dataSetHash %hash of dset object (dataset)
  MLEstate %optimal estimate of state (kalman-smoothed)
  causalMLE %optimal causal estimate (kalman-filtered)
  oneAheadMLE% optimal estimate one sample ahead (Kalman-predicted)
  logL
end

properties (Dependent)
  Nsamp
  bic
  aic
end
methods
  function this=dataFit(model,datSet,initC)
      this.model=model; %check that it is linsys
      this.dataSetHash=datSet.hash; %check that it is dset
      if nargin<3
        initC=[];
      end
      [MLE,filtered,oneAhead,rej,logL]=model.Ksmooth(datSet,initC);
      this.MLEstate=MLE;
      this.causalMLE=filtered;
      this.oneAheadMLE=oneAhead;
      this.logL=logL;
  end
  function Ns=get.Nsamp(this)
    Ns=this.causalMLE.Nsamp;
  end
  function bic=get.bic(this)
    bic=-2*this.logL+log(this.Nsamp)*this.model.dof;
  end
  function aic=get.aic(this)
    aic=-2*this.logL+2*this.model.dof;
  end
  function [p,chi]=likelihoodRatioTest(this,other)
    %Performs a likelihood ratio test for the fits of two models to a given dataset.
    %p returns the p-value of the observed likelihood ratio (of the model with more parameters over the one with less, so the ratio should always be larger than 1)
    %Models should be nested for this to work.
    %Check: both dataFits refer to the same dataset
    if this.dataSetHash ~= other.dataSetHash
      error('dataFit:LRT','Performing likelihood ratio test on fits of different datasets')
    end
    deltaDof=this.model.dof-other.model.dof;
    if deltaDof<0
      deltaDof=-deltaDof;
    end
    chi=2*(this.logL - other.logL);
    if chi<2
      error('dataFit:LRT','Model with more parameters has lower likelihood. This means either a bad fit or that models were not nested')
    end
    p=1-chi2cdf(chi,deltaDof);
  end
end

end
