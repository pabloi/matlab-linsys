%Test run:
observations=discretizeObs(round(rand(1,100)),2,[0,1]);
observations=discretizeObs(ones(1,100),2,[0,1]);
observationTimes=ceil([1:100]/10); %10 obs per time, can't be negative
pStateInitial=ones(101,1)/101; %Uniform prior, over 101 possible PSE values representing the [-200:200] mm/s range (4mm/s jumps)
bias=0;
sigma=75/1.1; %Realistic based on group-level analysis of responses
range=[-200:4:200];
p=1./(1+exp((range+bias)/sigma));
pObsGivenState=[p;1-p];
pStateGivenPrev=zeros(101,101)+diag(ones(101,1))+diag(ones(100,1),1)+diag(ones(100,1),-1); %States are assumed to move at most one state per step

%Inference:
[pPredicted, pUpdated, pSmoothed] = HMMstationaryInferenceAlt(observations,observationTimes,pObsGivenState,pStateGivenPrev,pStateInitial);

%Viz:
[fh] = vizHMMInference(pSmoothed,pStateGivenPrev,pObsGivenState,observations,observationTimes,range,[0 1],1:10);