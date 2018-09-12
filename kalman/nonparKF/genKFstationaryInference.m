function [pPredicted, pUpdated, pSmoothed] = genKFstationaryInference(observation,pObsGivenState,pStateGivenPrev,pStateInitial)
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
[D,M]=size(pObsGivenState); %M should equal N

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
pPredicted=nan(M,N+1); %We can predict up to the Nth+1 sample
pPredicted(:,1)=p0;
pUpdated=nan(M,N);
for i=1:N
   %Update:
   pUpdated(:,i) = genKFupdate(pPredicted(:,i),pObsGivenState(observation(i),:));
   %Predict:
   pPredicted(:,i+1) = genKFprediction(pUpdated(:,i),pStateGivenPrev);
end

if nargout>2 %Don't bother smoothing if user did not ask for it.
    %Backward pass (Smoothing)
    pSmoothed=nan(M,N);
    pSmoothed(:,N)=pUpdated(:,N);
    for i=(N-1):-1:1
        pSmoothed(:,i) = genKFsmooth(pSmoothed(:,i+1), pUpdated(:,i), pStateGivenPrev,pPredicted(:,i+1));
    end
end

end

function p=columnNormalize(p)
    %Normalization across columns
    p=p./sum(p,1);
end
function p=rowNormalize(p)
    %Normalization across columns
    p=p./sum(p,2);
end