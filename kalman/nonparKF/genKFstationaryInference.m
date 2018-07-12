function [pFiltered, pUpdated, pSmoothed] = genKFstationaryInference(observation,pObsGivenState,pStateGivenPrev,pStateInitial)
%Inference engine for a time-series of observations given stationary
%transition probabilities p(x_{k+1}|x_k) and observation probabilities
%p(y_k|x_k).
%INPUT:
%observations: an Nx1 vector of observations (N samples). Need to be an
%integer in the [1 M] range where M is size(pObsGivenPrev,1). Rows need
%to add to 1, if not they will be normalized.
%pObsGivenState: an MxD matrix that defines the observation probabilities
%given the state. D is the number of possible states. Columns need to add
%to 1, if not they will be normalized.
%pStateGivenPrev: DxD matrix containing transition probabilities
%p(x_{k+1}|x_k)
%pStateInitial: optional argument, 1xD vector defining initial state
%probabilities. Defaults to uniform distribution.

%Define relevant sizes:
N=length(observation);
[M,D]=size(pObsGivenState);

%Define init state if not given:
if nargin<4
    p0= ones(1,D)/D;%Uniform
else
    p0=pStateInitial;
end

%Check sizes:
%TO DO

%Normalize distributions, just in case. (I think this is cheaper than
%checking if they are normalized and normalizing only if they are not).
pObsGivenState=columnNormalize(pObsGivenState);
pStateGivenPrev=columnNormalize(pStateGivenPrev);

%Filter:
pFiltered=nan(N+1,D); %We can predict up to the Nth+1 sample
pFiltered(1,:)=p0;
pUpdated=nan(N,D);
for i=1:N
   %Update:
   pUpdated(i,:) = genKFupdate(pFiltered(i,:),pObsGivenState(observation(i),:));
   %Predict:
   pFiltered(i+1,:) = genKFprediction(pUpdated(i,:),pStateGivenPrev);
end

if nargout>2 %Don't bother smoothing if user did not ask for it.
    %Backward pass (Smoothing)
    pSmoothed=nan(N,D);
    pSmoothed(N,:)=pUpdated(N,:);
    for i=(N-1):-1:1
        pSmoothed(i,:) = genKFsmooth(pSmoothed(i+1,:), pUpdated(i,:), pStateGivenPrev,pFiltered(i+1,:));
    end
end

end

function p=columnNormalize(p)
    p=p./sum(p,1);
end