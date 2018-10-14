function [observationMatrix,transitionMatrix,stateDistrHistory]=HMM_EM(observationHistory,p0,O,T)

D=numel(p0);
[N,D1]=size(observationHistory);
%Init:
if nargin<3
observationMatrix=randn(D1,D);
else
  observationMatrix=O;
end
if nargin<4
  transitionMatrix=randn(D);
else
  transitionMatrix=T;
end
%Check: if observatinHistory is not an integer, discretize
%iterate
i=0;
while i<2%100
  %Estep:
  [~,~, stateDistrHistory] = HMMstationaryInference(observationHistory,observationMatrix,transitionMatrix,p0);
  %Mstep:
  [transitionMatrix,observationMatrix] = HMMmatrixEstim(stateDistrHistory,observationHistory);
  %Test for improvement:
  %logl=HMMlogL(observationHistory,observationMatrix,transitionMatrix,stateDistrHistory(1,:));
  i=i+1
end
end
