%% evaluate modelOrderSelection
%% Load data
%close all
clear all
load('./modelOrderTestS5RepsWVariableNoise.mat')
if ~exist('fitMdlFixedNoise') %5 reps requires renaming:
fitMdlFixedNoise=fitMdl;
end
%%
legacy_vizDataLikelihood(fitMdlFixedNoise,simDatSetFixedNoise) %Fixed noise
set(gcf,'Name','E-M fits to synthetic data from 4th order LTI-SSM')

%%
legacy_vizDataLikelihood(fitMdlVariableNoise,simDatSetVariableNoise) %Variable noise
set(gcf,'Name','E-M fits to synthetic data from 4th order LTI-SSM, with variable R')