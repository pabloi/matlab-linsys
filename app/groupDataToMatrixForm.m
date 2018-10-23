function [Y,Ysym,Ycom,U]=groupDataToMatrixForm(sqrtFlag,remove79flag)
%% Load real data:
load dynamicsData300blocksALTnormalization.mat
addpath(genpath('./fun/'))

% Some pre-proc
%B=nanmean(allDataEMG{1}(end-45:end-5,:,:)); %Baseline: last 40, exempting 5
clear data
if nargin>1 & remove79flag
  subjIdx=[2:6,8,10:16]; %No C01,C07,C09 so I can have 600 strides of Pos
else
  subjIdx=2:16; %Excluding C01 outlier only
end
muscPhaseIdx=1:360;
for i=1:5 %B,A1,A2,A3,P
    %Remove baseline
    data{i}=allDataEMG{i};
    %data{i}(end,:,:)=NaN; %Deleting last-stride of each subj

    %Interpolate over NaNs %This is only needed if we want to run fast
    %estimations, or if we want to avoid all subjects' data at one
    %timepoint from being discarded because of a single subject's missing
    %data
    %for j=1:size(data{i},3) %each subj
    %   t=1:size(data{i},1); nanidx=any(isnan(data{i}(:,:,j)),2); %Any muscle missing
    %   data{i}(:,:,j)=interp1(t(~nanidx),data{i}(~nanidx,:,j),t,'linear',0); %Substitute nans
    %end

    %Two subjects have less than 600 Post strides: C06, C08
    %Option 1: fill with zeros (current)
    %Option 2: remove them
    %Option 3: Use only 400 strides of POST, as those are common to all
    %subjects

    %Remove subj:
    data{i}=data{i}(:,muscPhaseIdx,subjIdx);
    if nargin>0
      %sum(data{i}(:)<0)
      %  data{i}(data{i}<0)=0;
      data{i}=sqrt(data{i});
    end

    %Compute asymmetry component
    aux=data{i}-fftshift(data{i},2);
    dataSym{i}=aux(:,1:size(aux,2)/2,:);
end

%% All data
%Y=cell2mat(cellfun(@(x) median(x,3),data','UniformOutput',false));
%U=[zeros(size(data{1},1),1);ones(sum(cellfun(@(x) size(x,1), data(2:4))),1);zeros(size(data{5},1),1);]';
%Ysym=cell2mat(cellfun(@(x) median(x-fftshift(x,2),3),data','UniformOutput',false));
%Ysym=Ysym(:,1:size(Ysym,2)/2);
%Ycom=cell2mat(cellfun(@(x) median(x+fftshift(x,2),3),data','UniformOutput',false));
%Ycom=Ycom(:,1:size(Ycom,2)/2);
Y=cell2mat(data');
U=[zeros(size(data{1},1),1);ones(sum(cellfun(@(x) size(x,1), data(2:4))),1);zeros(size(data{5},1),1);]';
Ysym=Y-fftshift(Y,2);
Ysym=Ysym(:,1:size(Ysym,2)/2,:);
Ycom=Y+fftshift(Y,2);
Ycom=Ycom(:,1:size(Ycom,2)/2,:);
end
