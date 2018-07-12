function [smoothedStateDistr] = genKFsmooth(nextStateSmoothedDistr,currStateDistr,nextStateGivenCurrDistr,nextStatePredictedDistr)
%Smoothing step in the Kalman smoother for generic (numeric)
%distributions
tol=1e-100;
smoothedStateDistr=currStateDistr .* sum(nextStateGivenCurrDistr .*(nextStateSmoothedDistr./(nextStatePredictedDistr+tol))',1);
end

