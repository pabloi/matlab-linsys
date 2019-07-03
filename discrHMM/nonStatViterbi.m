function [optSeq,logL]=nonStatViterbi(obs,transitionMatrix,emissionMatrix,priorP,input,obsTimes)
%Implementation of the Viterbi algorithm for non-stationary cases, where
%both the transition and emission matrices can change with a provided input
if nargin<6 || isempty(obsTimes)
    if ~numel(obs)==length(input)
        error('If observation time vector is not provided, obs must be a vector of the same length as input')
    end
end
error('This function has not been tested, try testNonStationaryHMM to debug first')

%Implements the Viterbi algorithm for finding the most likely state sequence given a sequence of observations
%The gist: for each point in time and each possible state, find the most likely
% sequence that would lead to EACH state at that time.
%This is easy to do through successive updates (dynamic programming) as the most
%likely sequence to go to any state can be computed by knowing the most likely
%sequence for each possible preceding state, the transition probability and the
% emission probabilities

N=length(input);

if ~isa(emissionMatrix,'function_handle')
    O=emissionMatrix;
else
    O=emissionMatrix(input(1));
end
if ~isa(transitionMatrix,'function_handle')
    T=transitionMatrix;
else
    T=transitionMatrix(input(1));
end
    O=columnNormalize(O);
lE=log(O)';  %Easier this way: accessing full columns
T=columnNormalize(T);
lT=log(T);

M=size(T,1);
if nargin<3
  warning('Prior not given, using uniform prior.')
  priorP=ones(1,M)/M; %Uniform prior if not given
else
    priorP=reshape(priorP,1,numel(priorP));
end
%Initialize variables:
%optimalSequence=nan(M,N);
optimalLogL=zeros(M,1); %Column
MLEprev=nan(M,N-1);

%Initialize with prior:
optimalLogL(:)=log(priorP);
aux=optimalLogL';

for i=1:N
    %Compute relevant emission matrix
    if isa(emissionMatrix,'function_handle') && (i==1 || input(i)~=input(i-1)) %Computing only if function, and only if input changed
        O=emissionMatrix(input(i));
        O=columnNormalize(O);
        lE=log(O)';
    end
    %Update using each observation for this time
    currentObs=obs(obsTimes==i);
    for j=1:length(currentObs)
        aux=aux+lE(:,currentObs(j));
    end
    
    if i>1
        [optimalLogL,MLEprev(:,i-1)]=max(aux,[],2); %Max,row-wise
    else
        [optimalLogL]=max(aux,[],2); %Max,row-wise
    end
    
    %Compute relevant transition matrix:
    if isa(transitionMatrix,'function_handle') && (i==1 || input(i)~=input(i-1)) %Computing only if function, and only if input changed
        T=transitionMatrix(input(i));
        T=columnNormalize(T);
        lT=log(T);
    end
   %Do transition
   aux=optimalLogL'+lT;
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
