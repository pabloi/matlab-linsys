function [updatedStateDistr] = genKFupdate(priorStateDistr,obsGivenStateDistr)
%Update step of kalman filter, accepting a generic discrete observation
%probability matrix. Implements: p(x_k|y_k) = p(y_k|x_k)p(x_k)/p(y_k)
%INPUT:
%priorStateDistr: p(x_k|{y_{k-1}}) [PRIOR STATE ESTIMATE]
%obsGivenStateDistr: p(y_k|x_k) [OBSERVATION MATRIX]
%OUTPUT:
%updatedStateDistr: p(x_k|y_k) [UPDATED STATE]

updatedStateDistr=obsGivenStateDistr'.*priorStateDistr;
s=sum(updatedStateDistr);
if s==0 %This is to avoid all 0 results if observation gets a 0 likelihood for possible states. The better solution is that this never happens. Issue warning?
    error('Obsevation has p=0 for all states with p>0. Impossible update. Winging it.')
end
%updatedStateDistr=updatedStateDistr/s; %Unnecessary if we only care about MAP
end
