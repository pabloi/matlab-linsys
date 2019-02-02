function [expandedModels,expandedInputs]=matchModelInputs(models,inputs)
  %This function takes a cell array of models (as defined by J,B,C,D,Q,R),
  %and re-defines B and D to support a larger input size by padding with 0's appropriately.
  %this is useful when two different models are fitted to explain the same output data from different number of INPUTS
  %INPUTS:
  %models: cell array of Models
  %inputs: cell array of inputs used for each model to describe the same dataset

  %First, determine unique number of inputs that exist:
  inputs=reshape(inputs,length(inputs),1);
  allInputs=cell2mat(inputs); %Cat-ting rows
  [expandedInputs,~,idxs]=unique(allInputs,'rows');
  Ninputs=size(expandedInputs,1);
  aux=cellfun(@(x) size(x,1),inputs);
  idxs=mat2cell(idxs,aux,1);
  %Then, re-assign models appropriately:
  expandedModels=models;
  for i=1:length(models)
      expandedModels{i}.B=zeros(size(models{i}.B,1),Ninputs);

      expandedModels{i}.B(:,idxs{i})=models{i}.B;
      expandedModels{i}.D=zeros(size(models{i}.D,1),Ninputs);
      expandedModels{i}.D(:,idxs{i})=models{i}.D;
  end
