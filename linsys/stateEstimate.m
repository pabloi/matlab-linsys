classdef stateEstimate

properties
  state=[]; %Mean of estimate distribution, N x 1 vector
  covar=[]; %Covariance matrix representing uncertainty of estimate, N x N matrix
  lagOneCovar=[]; %Covariance matrix of state estimate at time t with that of time t+1, optional
end
properties(Dependent)
  Nsamp
  order
  isMultiple
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
                          [~,D]=ldl(M);
                          if any(diag(D))<0
                            warning('stateEstimate:constructor','Uncertainty matrix is not PSD');
                          end
                      end
                  elseif size(P,3)==1
                      [~,D]=ldl(P);
                      if any(diag(D))<0
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
        initC=initCond(this.state(:,N),this.covar(:,:,N));
    end
    function fl=get.isMultiple(this)
        fl=iscell(this.state);
    end
    function newThis=extractSingle(this,i)
       if this.isMultiple
           if i>length(this.state)
               error('stateEstim:extractSingle',['Single index provided (' num2str(i) ') is larger than available number of states in this object (' num2str(length(this.state)) ').'])
           else
               newThis=stateEstimate(this.state{i},this.covar{i});
           end
       else
           error('stateEstim object is not multiple, cannot extract a single set')
       end
    end
    function newThis=marginalize(this,N)
        if isempty(this.lagOneCovar)
            newThis=stateEstimate(this.state(N,:),this.covar(N,N,:));
        else
            newLagOne=this.lagOneCovar(N,N,:);
            newThis=stateEstimate(this.state(N,:),this.covar(N,N,:),newLagOne);
        end
    end
    function plot(this,offset,prc,ph)
        
        if nargin<4
            ph=gca;
        end
        if nargin<3 || isempty(prc)
            prc=99.7; %99.7% percentile, roughly +- 3 std
        end
        if nargin<2
            offset=0;
        end
        axes(ph)
        hold on
        x=offset+[1:this.Nsamp];
        if prc>=0 && prc<=100
            f=norminv(1-.5*(1-prc/100),0,1); %Number of std away to get this percentile of the distribution in the shaded area
        else
            error('prc must be a number between 0 and 100')
        end
        for i=1:this.order
            set(gca,'ColorOrderIndex',i);
            y=this.state(i,:);
            pl=plot(x,y,'LineWidth',2);
            e=f*squeeze(sqrt(this.covar(i,i,:)))'; %To represent roughly a 99.7% CI (smaller CIs are usually hard to see)
            pp=patch([x fliplr(x)],[y+e, fliplr(y-e)],pl.Color,'EdgeColor','none','FaceAlpha',.5);
            uistack(pp,'bottom')
        end
    end
end
end
