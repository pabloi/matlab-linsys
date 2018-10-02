
%% Load real data:
load ../data/dynamicsData.mat
addpath(genpath('./fun/'))
% Some pre-proc
B=nanmean(allDataEMG{1}(end-45:end-5,:,:)); %Baseline: last 40, exempting 5
clear data dataSym
subjIdx=2:16;
%muscPhaseIdx=[1:(180-24),(180-11:180)];
%muscPhaseIdx=[muscPhaseIdx,muscPhaseIdx+180]; %Excluding PER
muscPhaseIdx=1:360;
for i=1:3 %B,A,P
    %Remove baseline
    data{i}=allDataEMG{i}-B;

    %Interpolate over NaNs %This is only needed if we want to run fast
    %estimations, or if we want to avoid all subjects' data at one
    %timepoint from being discarded because of a single subject's missing
    %data
    for j=1:size(data{i},3) %each subj
       t=1:size(data{i},1); nanidx=any(isnan(data{i}(:,:,j)),2); %Any muscle missing
       data{i}(:,:,j)=interp1(t(~nanidx),data{i}(~nanidx,:,j),t,'linear',0); %Substitute nans
    end
    
    %Two subjects have less than 600 Post strides: C06, C08
    %Option 1: fill with zeros (current)
    %Option 2: remove them
    %Option 3: Use only 400 strides of POST, as those are common to all
    %subjects
    
    %Remove subj:
    data{i}=data{i}(:,muscPhaseIdx,subjIdx);
    
    %Compute asymmetry component
    aux=data{i}-fftshift(data{i},2);
    dataSym{i}=aux(:,1:size(aux,2)/2,:);
    
end
%%
%% Define colormap:
ex1=[1,0,0];
ex2=[0,0,1];
mid=ones(1,3);
N=100;
map=[ex1.*[N:-1:1]'/N + mid.*[0:N-1]'/N; mid; ex2.*[0:N-1]'/N + mid.*[N:-1:1]'/N];
%% Some viz:
figure
for i=1:15
    subplot(4,4,i)
    surf(data{1}(:,:,i),'EdgeColor','none')
    colormap(flipud(map))
    caxis([-.5 .5])
end
%% Find bad:
for i=1:15
    for j=1:3 %Conditions
        bad=false(size(data{j}(:,:,i)));
        for k=1:360 %Muscle-phases
            dd=data{j}(:,k,i);
            dd=dd-median(dd);
            s=robCov(dd',90);
            z=sqrt((dd.^2)./s);
            bad(:,k)= z>5;
        end
    end
end