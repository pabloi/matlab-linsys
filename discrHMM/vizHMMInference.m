function [fh] = vizHMMInference(pEstimate,pStateGivenPrev,pObsGivenState,obs,obsTimes,stateRange,obsRange,timeRange)

if nargin<4
    stateRange=1:size(pEstimate,1);
    obsRange=1:size(pObsGivenState,1);
end
if nargin<6
timeRange=1:size(pEstimate,2);
end
timeRange=timeRange(:);

fh=figure('Units','Pixels','InnerPosition',[100 100 300*4 300*2]);
%subplot(3,2,1) %Transition matrix
ax=axes;
ax.Position=[.1 .69 .39 .25]; 
cc=.7*get(gca,'ColorOrder');
t=[0:100]'/100;
map=ones(1,3).*(1-t)+t.*cc(1,:);
imagesc(stateRange,stateRange,columnNormalize(pStateGivenPrev))
title('Transition matrix')
axis tight
ax=gca;
ax.YAxis.Direction='normal';
ax.XAxis.Direction='normal';
xlabel('Current state')
ylabel('Next state')
colormap(map)
ax.CLim=[0 max(pStateGivenPrev(:))];
colorbar


%subplot(3,2,2) %Obs matrix
ax=axes;
ax.Position=[.58 .69 .39 .25]; 
imagesc(stateRange,obsRange,pObsGivenState)
title('Observation matrix')
axis tight
ax=gca;
ax.YAxis.Direction='normal';
ax.XAxis.Direction='normal';
xlabel('State')
ylabel('Obs')
colormap(map)
ax.CLim=[0 max(pObsGivenState(:))];
colorbar

%subplot(3,2,3:4) %Estimate evolution
ax=axes;
ax.Position=[.1 .34 .85 .25]; 
imagesc(timeRange,stateRange,pEstimate)
ax=gca;
ax.YAxis.Direction='normal';
ax.XAxis.Direction='normal';
hold on
[~,MLE]=max(pEstimate);
plot(timeRange,range(MLE),'r','LineWidth',2)
axis tight
colormap(map)
ax.CLim=[0 max(pEstimate(:))];
colorbar

%subplot(3,2,5:6) %Observations
ax=axes;
ax.Position=[.1 .04 .85 .25]; 
%obsPrediction= ;% Estimate of predicted responses at each point in time, if there are many responses at a single point, then the average response estimate is plotted
%imagesc(obsPrediction)
hold on
scatter(timeRange(obsTimes)+.5*(randn(size(obsTimes))-.5),obsRange(obs),'filled')
%To do: something to separate multipe obs at same time visually

end

