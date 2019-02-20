%% Testing functionality of dset
load ../sampleObjects.mat
%% Create
this = dset(datSet.in,datSet.out); %Identical

%% Split:
breaks=find(rand(1,size(datSet.in,2))<.003);
multiSet=datSet.split(breaks);

%% Viz
fh=datSet.vizFit(model) %Single model
fh=datSet.vizFit({model,model}) %Two (identic) models

fh=datSet.vizRes(model)
fh=datSet.vizRes({model,model})

fh=datSet.compareModels({model,model}) %Need at least two
