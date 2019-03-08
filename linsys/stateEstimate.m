classdef stateEstimate

properties
  state=[]; %Mean of estimate distribution, N x 1 vector
  covar=[]; %Covariance matrix representing uncertainty of estimate, N x N matrix
  lagOneCovar=[]; %Covariance matrix of state estimate at time t with that of time t+1, optional
end
properties(Dependent)
  Nsamp
  order
end
methods
    function this=stateEstimate(x,P,Plag)
        if nargin>0
          this.state=x;
          if nargin>1
              if size(x,1)~=size(P,1)
                error('stateEstimate:constructor','Inconsistent state and uncertainty sizes')
              end
              if size(P,1)~=size(P,2)
                  error('stateEstimate:constructor','Uncertainty matrix is not square')
              end
              %Need to check: P is psd for each sample
              if ~isempty(P)
                  if size(P,3)==size(x,2)
                      this.covar=P;
                      for i=1:size(P,3)
                          M=P(:,:,i);
                          try
                            cM=chol(M); %This may fail if not psd
                          catch
                            warning('stateEstimate:constructor','Uncertainty matrix is not PSD');
                          end
                      end
                  elseif size(P,3)==1
                     try 
                         chol(P);
                     catch
                         warning('stateEstimate:constructor','Uncertainty matrix is not PSD');
                     end
                     this.covar=repmat(P,1,1,size(x,2));
                  else
                     error('stateEstimateConstructor:inconsistentCovarSize',...
                     'Provided covariance matrix does not have the same number of samples as the state'); 
                  end
              end
          end
        end
      
      if nargin>2
        this.lagOneCovar=Plag;
        if size(Plag,3)~=size(P,3)-1
          error('Inconsistent covariance matrix sizes')
        end
      end
    end
    function Ns=get.Nsamp(this)
      Ns=size(this.state,2);
    end
    function ord=get.order(this)
      ord=size(this.state,1);
    end
    function initC=getSample(this,N)
      %Gets a single sample to be used as initial condition in other functions
        initC=initCond(this.x(:,N),this.P(:,:,N));
    end

end
end