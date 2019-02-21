%% evaluate modelOrderSelection
%% Load data
load('./modelOrderTestS5RepsWVariableNoise.mat')

%%
vizDataLikelihood(fitMdl,simDatSetFixedNoise) %Fixed noise

vizDataLikelihood(fitMdlVariableNoise,simDatSetVariableNoise) %Fixed noise