function seeModels(models,Y,U)

fh=figure;
nD=size(models{1}.X,1);
for i=1:length(models)
   [Xs,Ps,Pt,Xf,Pf,Xp,Pp,rejSamples]=statKalmanSmoother(Y,models{i}.J,models{i}.C,models{i}.Q,models{i}.R,models{i}.X(:,1),models{i}.P(:,:,1),models{i}.B,models{i}.D,U,false,true); 
   models{i}.Xs=Xs; %Smoothed data
   models{i}.Xf=Xf; %Filtered data
   [Xs,Ps,Pt,Xf,Pf,Xp,Pp,rejSamples]=statKalmanSmoother(Y,models{i}.J,models{i}.C,models{i}.Q,models{i}.R,models{i}.X(:,1),models{i}.P(:,:,1),models{i}.B,models{i}.D,U,true,true); 
   models{i}.Xfrob=Xf; %Filtered data with outliers
   models{i}.dataProj2=models{i}.C\(Y-models{i}.D*U); %Simply projecting data over state space
   models{i}.dataProj=(models{i}.C'*inv(models{i}.R)*models{i}.C)\(models{i}.C'*(models{i}.R\(Y-models{i}.D*U))); %Simply projecting data over state space
end
for k=1:nD
    subplot(nD,1,k)
    hold on
    for i=1:length(models)
        scatter(1:length(rejSamples),models{i}.dataProj2(k,:),10,.6*ones(1,3),'filled','DisplayName','Data projected alt')
        scatter(find(rejSamples),models{i}.dataProj(k,rejSamples),30,'filled','r','DisplayName','Data rejected')
        scatter(1:length(rejSamples),models{i}.dataProj(k,:),10,'filled','k','DisplayName','Data projected')
        %scatter(find(rejSamples),models{i}.dataProj2(k,rejSamples),30,'filled','b','DisplayName','Data rejected')
        
        plot(models{i}.Xs(k,:),'DisplayName','Smooth')
        plot(models{i}.Xf(k,:),'DisplayName','Filtered')
        plot(models{i}.Xfrob(k,:),'DisplayName','Filtered robust')
    end
    legend('Location','SouthEast')
end
    
end