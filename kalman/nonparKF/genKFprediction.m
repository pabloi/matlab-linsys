function predictedStateDistr = genKFprediction(priorStateDistr,nextStateGivenCurrDistr)
%Prediction step update of numerical kalman filter.
%Implements: p(x_{k+1}|)=\int p(x_{k+1}|x_k)p(x_k) dx_k
%INPUT
%nextStateGivenCurrDistr: p(x_{k+1}|x_k) [TRANSITION MATRIX]
%priorStateDistr: p(x_k) [PRIOR STATE ESTIMATE]
%OUTPUT:
%predictedStateDistr: p(x_{k+1}) [PREDICTED STATE]

predictedStateDistr=sum(nextStateGivenCurrDistr .* priorStateDistr,2)';
end
