%% evaluate modelOrderSelection
%% Load data
load('./modelOrderTestS5RepsWVariableNoise.mat')

%%
legacy_vizDataLikelihood(fitMdl,simDatSetFixedNoise) %Fixed noise
set(gcf,'Name','E-M fits to synthetic data from 4th order LTI-SSM')

%%
vizDataLikelihood(fitMdlVariableNoise,simDatSetVariableNoise) %Fixed noise
set(gcf,'Name','E-M fits to synthetic data from 4th order LTI-SSM, with variable R')