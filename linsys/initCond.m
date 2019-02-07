classdef initCond
    %INITCOND Class representing a set of initial condition (gaussian) beliefs for
    %bayesian-like filtering. To be used with linsys' kalman filtering and
    %smoothing.
    
    properties
       x %State belief: N x 1 vector
       P %State uncertainty: N x N matrix
    end
    properties(Dependent)
       order
    end
    
    methods
        function obj = initCond(x,P)
            obj.x=reshape(x,numel(x),1);
            obj.P=P;
            if numel(x)~=size(P,1)
                error('initCond:constructor','Inconsistent state and uncertainty sizes')
            end
            if size(P,1)~=size(P,2)
                error('initCond:constructor','Uncertainty matrix is not square')
            end
            if ~isempty(P) && (min(eig(P))<0 || any(imag(eig(P))~=0))
                error('initCond:constructor','Uncertainty matrix is not PSD')
            end
        end
        function order=get.order(this)
           order=length(this.x); 
        end
    end
end

