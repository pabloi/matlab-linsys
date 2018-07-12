function [updatedStateDistr] = genKFupdate(priorStateDistr,obsGivenStateDistr)
%Update step of kalman filter, accepting a generic discrete observation
%probability matrix
updatedStateDistr=normalize(obsGivenStateDistr.*priorStateDistr);
end

function p=normalize(p)
    p=p/sum(p(:));
end

