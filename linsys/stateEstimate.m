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
            this.covar=P;
              if size(x,1)~=size(P,1)
                error('stateEstimate:constructor','Inconsistent state and uncertainty sizes')
              end
              if size(P,1)~=size(P,2)
                  error('stateEstimate:constructor','Uncertainty matrix is not square')
              end
              %Need to check: P is psd for each sample
              if ~isempty(P)
              for i=1:size(P,3)
              if ~any(isinf(diag(P(:,:,i)))) && (min(eig(P(:,:,i)))<0 || any(imag(eig(P(:,:,i)))~=0))
                  error('stateEstimate:constructor','Uncertainty matrix is not PSD')
              end
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
