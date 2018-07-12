function predictedStateDistr = genKFprediction(priorStateDistr,nextStateGivenCurrDistr)
%Prediction step update of numerical kalman filter
predictedStateDistr=sum(nextStateGivenCurrDistr .* priorStateDistr,2)';
end

