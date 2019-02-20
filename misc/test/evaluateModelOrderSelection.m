%% evaluate modelOrderSelection
%% Load data
load('/Datos/Documentos/code/matlab-linsys/misc/test/modelOrderTestS5RepsWVariableNoise.mat')

%%
vizDataLikelihood(fitMdl,simDatSetFixedNoise) %Fixed noise

vizDataLikelihood(fitMdlVariableNoise,simDatSetVariableNoise) %Fixed noise