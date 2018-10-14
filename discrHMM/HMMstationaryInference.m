function [pPredicted, pUpdated, pSmoothed] = HMMstationaryInference(observation,pObsGivenState,pStateGivenPrev,pStateInitial)
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
pPredicted=nan(M,N+1); %We can predict up to the Nth+1 sample, should sparsify
p=p0;
pPredicted(:,1)=p;
pUpdated=nan(M,N); %Should sparsify
for i=1:N
   %Update:
   p = HMMupdate(p,pObsGivenState(observation(i),:));
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
    pSmoothed=nan(M,N); %Should sparsify
    p=pUpdated(:,N);
    pSmoothed(:,N)=p;
    for i=(N-1):-1:1
        p = HMMbackUpdate(p, pUpdated(:,i), pStateGivenPrev,pPredicted(:,i+1));
        pSmoothed(:,i) = p;
    end
end

end

function p=columnNormalize(p)
    %Normalization across columns
    p=p./sum(p,1);
end
