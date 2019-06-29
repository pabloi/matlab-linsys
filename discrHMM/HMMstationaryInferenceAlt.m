function [pPredicted, pUpdated, pSmoothed] = HMMstationaryInferenceAlt(observations,observationTimes,pObsGivenState,pStateGivenPrev,pStateInitial)
%Modified version of HMMstationaryInference to allow for arbitrary
%observation times, and multiple observations for single time.
%Observations and observationTimes must be same length, both discrete.
%From HMMstationaryInference:
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
%Functionally equivalent to the forward-backward algorithm, although implemented
%in the kalman-smoother style

%Sort observations by time:
[obsTimes,idx]=sort(observationTimes,'ascend');
endT=obsTimes(end);
if obsTimes(1)<1
    error('Observation times before initial condition!')
end
startT=1; %starting time is always assumed to be 1 (corresponding to pStateInitial)
observations=observations(idx);

%Check that obs and obsTimes are discrete
%To do.

%Define relevant sizes:
N=length(observations);
[D,M]=size(pObsGivenState); %M should equal N

%Define init state if not given:
if nargin<4
    p0= ones(D,1)/D;%Uniform
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
pPredicted=nan(M,endT+1); %We can predict up to the Nth+1 sample, should sparsify
p=p0;
pPredicted(:,1)=p;
pUpdated=nan(M,endT); %Should sparsify
for i=startT:endT
   %Update:
   currentTimeObs=observations(observationTimes==i);
   for j=1:length(currentTimeObs)
    p = HMMupdate(p,pObsGivenState(currentTimeObs(j),:));
   end
    pUpdated(:,i) = p;
   %Predict:
   p = HMMpredict(p,pStateGivenPrev);
   pPredicted(:,i+1) = p;
   if mod(i,1)==0
       p=p/sum(p); %Normalization is needed ocassionally to prevent underflow of all states
   end
end

if nargout>2 %Don't bother smoothing if user did not ask for it.
    %Backward pass (Smoothing)
    pSmoothed=nan(M,endT); %Should sparsify
    p=pUpdated(:,endT);
    pSmoothed(:,endT)=p;
    for i=(endT-1):-1:startT
        p = HMMbackUpdate(p, pUpdated(:,i), pStateGivenPrev,pPredicted(:,i+1));
        pSmoothed(:,i) = p;
    end
end

end

function p=columnNormalize(p)
    %Normalization across columns
    p=p./sum(p,1);
end
