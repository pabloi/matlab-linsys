%% Script to test which of the model order selection criteria works best in practice
addpath(genpath('../../'))
%% Step 1: define high-dim model
load ../../../EMG-LTI-SSM/res/allDataModels.mat
model=model{5}; %4-th order model as ground truth
initC=initCond(zeros(4,1),zeros(4));
deterministicFlag=false;

%% Step 2: for single-rep estimation (PCA+fast+ refinement stage), run sim and estimate
reps=1e2;
for i=1:reps
  %Simulate realization:
  [simDatSet{i},stateE]=model.simulate(datSet.in,initC,deterministicFlag);
  %Estimate:
  [fitMdl{i},outlog{i}]=linsys.id(simDatSet{i},4);
end
save modelVsrTestSingleRep.mat fitMdl outlog simDatSet datSet model
%% Step 3: same thing, but with 5 reps
opts.Nreps=5;
reps=1e2;
for i=1:reps
  %Simulate realization:
  [simDatSet{i},stateE]=model.simulate(datSet.in,initC,deterministicFlag);
  %Estimate:
  [fitMdl{i},outlog{i}]=linsys.id(simDatSet{i},4,opts);
end
save modelVsrTest5Reps.mat fitMdl outlog simDatSet datSet model

%% View time constants:
tc=cell2mat(cellfun(@(x) eig(x.A),fitMdl,'UniformOutput',false)); %Get time constants
trueTC=sort(eig(model.A));
%Make time constants real:
rtc=real(tc);
rtc=sort(rtc);

%See dispersion:
figure; hold on;
for i=1:4
    histogram(rtc(i,:))
    plot(trueTC(i)*[1 1],[0 40],'k--') %True value
end
