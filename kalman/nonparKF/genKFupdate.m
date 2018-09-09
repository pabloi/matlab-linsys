function [updatedStateDistr] = genKFupdate(priorStateDistr,obsGivenStateDistr)
%Update step of kalman filter, accepting a generic discrete observation
%probability matrix. Implements: p(x_k|y_k) = p(y_k|x_k)p(x_k)/p(y_k)
%INPUT:
%priorStateDistr: p(x_k|{y_{k-1}}) [PRIOR STATE ESTIMATE]
%obsGivenStateDistr: p(y_k|x_k) [OBSERVATION MATRIX]
%OUTPUT:
%updatedStateDistr: p(x_k|y_k) [UPDATED STATE]

updatedStateDistr=normalize(obsGivenStateDistr.*priorStateDistr);
end

function p=normalize(p)
    p=p/sum(p(:));
end
