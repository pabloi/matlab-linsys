function genEM(observationHistory,p0)

D=numel(p0);
[N,D1]=size(observationHistory);
%Init:
observationMatrix=randn(D1,D);
transitionMatrix=randn(D);

%iterate
while
  %Estep:
  [~,~, stateDistrHistory] = genKFstationaryInference(observation,observationMatrix,transitionMatrix,p0);
  %Mstep:
  [transitionMatrix,observationMatrix] = genParamEstim(stateDistrHistory,observationHistory);
  %Test for improvement:
  logl=genLogL(observationHistory,observationMatrix,transitionMatrix,stateDistrHistory(1,:));
end
end
