classdef dataFit
%This is just a collection of a model, a dataset, and a state estimate to implement some useful functions
properties (SetAccess = immutable)
  model
  dataSet
  stateEstim
  initialCondition=initCond([],[]);
  fitMethod='KS';
  goodnessOfFit %Metric to quantify the fit, can be logL, RMSE depending on how data was fit
end

properties (Dependent)
  output
  residual
end
methods
  function this=dataFit(model,datSet,fitMethod,initC)
      if nargin<4
        initC=initCond([],[]);
      end
      if nargin<3 || isempty(fitMethod)
        fitMethod='KS';
      end
      [MLE,filtered,oneAhead,rej,logL]=model.Ksmooth(datSet,initC);
      switch fitMethod
        case 'KF' %Kalman filter
          this.stateEstim=filtered;
        case 'KS' %Kalman smoother
          this.stateEstim=MLE;
        case 'KP' %Kalman one-ahead predictor
          this.stateEstim=oneAhead;
      end
      this.model=model;
      this.dataSet=datSet;
      this.fitMethod=fitMethod;
      this.goodnessOfFit=logL;
  end
  function res=get.residual(this)
      res=this.output - this.dataSet.out;
  end
  function out=get.output(this)
      out=this.model.C*this.stateEstim.state(:,1:end-1)+this.model.D*this.dataSet.in;
  end
end

end
