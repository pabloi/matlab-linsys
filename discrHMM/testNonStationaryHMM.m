input=[zeros(1,10),20*ones(1,10),zeros(1,10)];
N=length(input);
bias=0;
sigma=75/1.1; %Realistic based on group-level analysis of responses
range=[-200:4:200];
p=1./(1+exp((range+bias)/sigma));
pObsGivenState=[p;1-p];

transitionWidth=15;
R=[ones(1,transitionWidth)./[1:transitionWidth],zeros(1,101-transitionWidth)];
p1=toeplitz(R,R'); %Exponentiallly decaying transition probabilities to adjacent states
pStateGivenPrev=@(u) p1; %Stationary transition matrix, should also work (the estimation will only be good if the transition band is wide enough to allow for the non-stationarities in the data to be interpreted as noise)
pStateGivenPrev=@(u) conv2(p1,[(u<0)*ones(abs(u),1)./[abs(u):-1:1]';1; (u>0)*ones(abs(u),1)./[1:abs(u)]'],'same'); %The more positive u, the more likely an upward state transition is

underlyingState=[-75*ones(1,10),[-75:15:74],75*ones(1,10)]; 
observationTimes=sort(randi(N,1,300),'ascend'); %300 observations at 30 random points in time (not many!)
underlyingObsP=1./(1+exp((bias-underlyingState(observationTimes))/sigma)); %Probability of emission at each time
observations=discretizeObs(binornd(1,underlyingObsP),2,[0,1]);

%Inference:
[pPredicted, pUpdated, pSmoothed] = HMMnonStationaryInferenceAlt(observations,observationTimes,input,pObsGivenState,pStateGivenPrev,pStateInitial);

%Viz:
[fh] = vizHMMInference(pSmoothed,pStateGivenPrev(0),pObsGivenState,observations,observationTimes,range,[0 1],1:N);

%Add true state:
ph=findobj(fh,'Type','Axes');
axes(ph(2))
plot(1:N,underlyingState,'k','LineWidth',1)

%Add avg. observation:
axes(ph(1))
hold on
aux=splitapply(@(x) sum(x==2)/length(x), observations, observationTimes);
plot(1:N,aux,'r')

%Add viterbi sequence: %TO DO
%[optSeq,logL]=viterbi(obs,transitionMatrix,emissionMatrix,priorP);
%%Viterbi is not implemented to have multiple obs at a single time, nor
%%times with no obs.
%plot(1:N,optSeq,'r')