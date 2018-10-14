function logl=genLogL(observationHistory,observationMatrix,transitionMatrix,p0)

%Kalman Filter:
[pFiltered] = genKFstationaryInference(observationHistory,observationMatrix,transitionMatrix,p0);

%Compute likrlihood:

end
