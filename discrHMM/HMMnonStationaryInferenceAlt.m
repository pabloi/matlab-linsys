function [pPredicted, pUpdated, pSmoothed] = HMMnonStationaryInferenceAlt(observations,observationTimes,input,pObsGivenState,pStateGivenPrev,pStateInitial)
%Modified version of HMMstationaryInferenceAlt to allow for an input that
%determines the matrices pObsGivenState and pStateGivenPrev (they can still
%be matrices too).
%Eventually this will supersede HMMstationaryInferenceAlt
%
%From HMMstationaryInferenceAlt:
%Modified version of HMMstationaryInference to allow for arbitrary
%observation times, and multiple observations for single time.
%Observations and observationTimes must be same length, both discrete.
%
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
[observationTimes,idx]=sort(observationTimes,'ascend');
endT=length(input);
if observationTimes(1)<1
    error('Observation times before initial condition!')
end
if observationTimes(end)>endT
    error('Observations outside the input range')
end
startT=1; %starting time is always assumed to be 1 (corresponding to pStateInitial)
observations=observations(idx);

%Check that obs and obsTimes are discrete
%To do.

%Define relevant sizes:
N=length(observations);
if ~isa(pObsGivenState,'function_handle')
    O=pObsGivenState;
    O=columnNormalize(O);
else
    O=pObsGivenState(0);
end
[D,M]=size(O); %M should equal N

%Define init state if not given:
if nargin<4
    p0= ones(D,1)/D;%Uniform
else
    p0=pStateInitial;
end

%Check sizes:
%TO DO

%Filter:
pPredicted=nan(M,endT+1); %We can predict up to the Nth+1 sample, should sparsify
p=p0;
pPredicted(:,1)=p;
pUpdated=nan(M,endT); %Should sparsify
if ~isa(pObsGivenState,'function_handle')
    O=pObsGivenState;
    O=columnNormalize(O);
end
if ~isa(pStateGivenPrev,'function_handle')
    T=pStateGivenPrev;
    T=columnNormalize(T);
end
for i=startT:endT
    %Compute matrices if necessary:
    if isa(pObsGivenState,'function_handle') && (i==startT || input(i)~=input(i-1)) %Computing only if function, and only if input changed
        O=pObsGivenState(input(i));
        O=columnNormalize(O);
    end
    if isa(pStateGivenPrev,'function_handle') && (i==startT || input(i)~=input(i-1)) %Computing only if function, and only if input changed
        T=pStateGivenPrev(input(i));
        T=columnNormalize(T);
    end
   %Update:
   currentTimeObs=observations(observationTimes==i);
   if ~isempty(currentTimeObs)
       1;
   for j=1:length(currentTimeObs)
    p = HMMupdate(p,O(currentTimeObs(j),:));
   end
   end
    pUpdated(:,i) = p;
   %Predict:
   p = HMMpredict(p,T);
   pPredicted(:,i+1) = p;
   if mod(i,1)==0
       p=p/sum(p); %Normalization is needed ocassionally to prevent underflow of all states
   end
end
pUpdated=columnNormalize(pUpdated);
pPredicted=columnNormalize(pPredicted);

if nargout>2 %Don't bother smoothing if user did not ask for it.
    %Backward pass (Smoothing)
    pSmoothed=nan(M,endT); %Should sparsify
    p=pUpdated(:,endT);
    pSmoothed(:,endT)=p;
    for i=(endT-1):-1:startT
        if isa(pStateGivenPrev,'function_handle') && (i==(endT-1) || input(i)~=input(i+1)) %Computing only if function, and only if input changed
            T=pStateGivenPrev(input(i));
        end
        p = HMMbackUpdate(p, pUpdated(:,i), T,pPredicted(:,i+1));
        pSmoothed(:,i) = p;
    end
    pSmoothed=columnNormalize(pSmoothed);
end

end

function p=columnNormalize(p)
    %Normalization across columns
    p=p./sum(p,1);
end
