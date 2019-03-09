classdef initCond < stateEstimate
    %INITCOND Class representing a set of initial condition (gaussian) beliefs for
    %bayesian-like filtering. To be used with linsys' kalman filtering and
    %smoothing. Inherits from stateEstimate class, but represents a single time point.

    methods
        function obj = initCond(x,P)
          if size(x,2)>1
            error('Initial condition estimates must represent a single time-sample')
          elseif isempty(x)
              warning('Empty initial condition given. Using improper prior.')
              N=size(x,1);
              x=zeros(N,1);
              P=diag(Inf*ones(N,1));
          end
          obj.state=x;
          obj.covar=P;
        end
    end
end
