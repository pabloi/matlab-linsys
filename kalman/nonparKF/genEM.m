function [observationMatrix,transitionMatrix]=genEM(observationHistory,p0,O,T)

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
while i<1%100
  %Estep:
  [~,~, stateDistrHistory] = genKFstationaryInference(observationHistory,observationMatrix,transitionMatrix,p0);
  %Mstep:
  [transitionMatrix,observationMatrix] = genParamEstim(stateDistrHistory,observationHistory);
  %Test for improvement:
  %logl=genLogL(observationHistory,observationMatrix,transitionMatrix,stateDistrHistory(1,:));
  i=i+1
end
end
