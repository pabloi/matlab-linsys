function [optSeq,logL]=viterbi(obs,transitionMatrix,emissionMatrix,priorP)
%Implements the Viterbi algorithm for finding the most likely state sequence given a sequence of observations
%The gist: for each point in time and each possible state, find the most likely
% sequence that would lead to EACH state at that time.
%This is easy to do through successive updates (dynamic programming) as the most
%likely sequence to go to any state can be computed by knowing the most likely
%sequence for each possible preceding state, the transition probability and the
% emission probabilities

%TO DO: viterbi is highly parallelizable, should do it for improved performance (GPU implementation? simple parfor?)
%TO DO: lazy viterbi
N=numel(obs);
M=size(transitionMatrix,1);
emissionMatrix=columnNormalize(emissionMatrix);
transitionMatrix=columnNormalize(transitionMatrix);
if nargin<3
  warning('Prior not given, using uniform prior.')
  priorP=ones(1,M)/M; %Uniform prior if not given
end
%Initialize variables:
%optimalSequence=nan(M,N);
optimalLogL=zeros(M,1); %Column
MLEprev=nan(M,N-1);
lE=log(emissionMatrix)';  %Easier this way: accessing full columns
lT=log(transitionMatrix);

%Step 1:
optimalLogL(:)=log(priorP)+lE(:,obs(1));

for i=2:N
  aux=optimalLogL'+lT+lE(:,obs(i));
  [optimalLogL,MLEprev(:,i-1)]=max(aux,[],2); %Max,row-wise
end
optSeq=nan(1,N);
[logL,optSeq(end)]=max(optimalLogL); %Most likely final state
nextState=optSeq(end);
for k=(N-1):-1:1
  nextState=MLEprev(nextState,k);
  optSeq(k)=nextState;
end

end

function p=columnNormalize(p)
    %Normalization across columns
    p=p./sum(p,1);
end
