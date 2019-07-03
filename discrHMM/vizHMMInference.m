function [fh] = vizHMMInference(pEstimate,pStateGivenPrev,pObsGivenState,obs,obsTimes,stateRange,obsRange,timeRange)

if nargin<4
    stateRange=1:size(pEstimate,1);
    obsRange=1:size(pObsGivenState,1);
end
if nargin<6
timeRange=1:size(pEstimate,2);
end

fh=figure;
subplot(3,2,1) %Transition matrix
cc=get(gca,'ColorOrder');
t=[0:100]'/100;
map=ones(1,3).*(1-t)+t.*cc(1,:);
imagesc(stateRange,stateRange,pStateGivenPrev)
title('Transition matrix')
axis tight
ax=gca;
ax.YAxis.Direction='normal';
ax.XAxis.Direction='normal';
xlabel('Current state')
ylabel('Next state')
colormap(map)
ax.CLim=[0 max(pStateGivenPrev(:))];


subplot(3,2,2) %Obs matrix
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

subplot(3,2,3:4) %Estimate evolution
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

subplot(3,2,5:6) %Observations
scatter(timeRange(obsTimes)+.5*(randn(size(obsTimes))-.5),obsRange(obs),'filled')
%To do: something to separate multipe obs at same time visually

end

