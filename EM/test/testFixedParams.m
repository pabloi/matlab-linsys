%% Load order 4 model
clear all
load ../../misc/refData/refModelOrder4.mat mdl datSet

%% Generate simuluation
initC=initCond(zeros(4,1),zeros(4));
deterministicFlag=false;
[simDatSet,stateE]=mdl.simulate(datSet.in,initC,deterministicFlag);

%% Fit model with all free params
opts.Nreps=1;
opts.fastFlag=0; %Slow mode for odd/even CV, because of NaN data
opts.indB=1;
opts.indD=[];
allFreeModel=linsys.id(simDatSet,4,opts);
allFreeModel.name='all free';

%% Fit model with fixed A
opts.fixA=allFreeModel.A;
fixedAmodel=linsys.id(simDatSet,4,opts);
fixedAmodel.name='fixed A';

%% Fit model with fixed B
opts.fixA=[];
opts.fixB=allFreeModel.B;
fixedBmodel=linsys.id(simDatSet,4,opts);
fixedBmodel.name='fixed B';

%% Fit model with fixed C
opts.fixB=[];
opts.fixC=allFreeModel.C;
fixedCmodel=linsys.id(simDatSet,4,opts);
fixedCmodel.name='fixed C';

%% Fit model with fixed D
opts.fixC=[];
opts.fixD=allFreeModel.D;
fixedDmodel=linsys.id(simDatSet,4,opts);
fixedDmodel.name='fixed D';

%% Fit model with fixed Q
opts.fixD=[];
opts.fixQ=allFreeModel.Q;
fixedQmodel=linsys.id(simDatSet,4,opts);
fixedQmodel.name='fixed Q';

%% Fit model with fixed R
opts.fixQ=[];
opts.fixR=allFreeModel.R;
fixedRmodel=linsys.id(simDatSet,4,opts);
fixedRmodel.name='fixed R';

%% Fit model with fixed x0
opts.fixR=[];
opts.fixX0=zeros(size(mdl.A,1),1);
fixedXmodel=linsys.id(simDatSet,4,opts);
fixedXmodel.name='fixed x0';

%% Fit model with fixed P0
opts.fixX0=[];
opts.fixP0=eye(size(mdl.A));
fixedPmodel=linsys.id(simDatSet,4,opts);
fixedPmodel.name='fixed P0';

%% Fit diagonal A
opts.fixP0=[];
opts.diagA=true;
diagAmodel=linsys.id(simDatSet,4,opts);
diagAmodel.name='diag A';

%% Fit fixed B,C (Randomly selected pair of params)
opts.diagA=false;
opts.fixB=allFreeModel.B;
opts.fixC=allFreeModel.C;
fixedBCmodel=linsys.id(simDatSet,4,opts);
fixedBCmodel.name='fixed B,C';

%%
save testFixedParamsResults.mat fixed* mdl datSet simDatSet allFreeModel

%%
allMdl={fixedAmodel fixedBmodel fixedCmodel fixedDmodel fixedQmodel fixedRmodel fixedXmodel fixedPmodel allFreeModel diagAmodel fixedBCmodel};
allMdl=cellfun(@(x) x.canonize('canonical'),allMdl,'UniformOutput',false);
vizDataLikelihood(allMdl,simDatSet)
