function [fh] = vizModels(model)

M=size(model{1}.X,1);
fh=figure('Units','Normalized','OuterPosition',[0 0 1 1]);
Ny=6;
Nx=M+3;

%% Compute output and residuals
U=[zeros(1,100) ones(1,1000)];
for i=1:length(model)
    [Y2,X2]=fwdSim(U,model{i}.J,model{i}.B,model{i}.C,model{i}.D,[],[],[]);
    model{i}.smoothStates=X2;
    model{i}.smoothOut=Y2;
end
%% Plot STATES
for i=1:M
subplot(Nx,Ny,Ny*(i-1)+[1:2]) %States
hold on
%Smooth states
set(gca,'ColorOrderIndex',1)
clear p
for k=1:length(model)
    p(k)=plot(model{k}.smoothStates(i,:),'LineWidth',2,'DisplayName',[model{k}.name ', \tau=' num2str(-1./log(model{k}.J(i,i)),3)]);
end
title('(Smoothed) Step-response states')
%if i==1
legend(p,'Location','SouthEast')
%end
ylabel(['State ' num2str(i)])
end
    
%% Define colormap:
ex1=[1,0,0];
ex2=[0,0,1];
mid=ones(1,3);
N=100;
map=[ex1.*[N:-1:1]'/N + mid.*[0:N-1]'/N; mid; ex2.*[0:N-1]'/N + mid.*[N:-1:1]'/N];
%% Plot C and D columns
aC=max(abs(model{1}.C(:)));
for i=1:length(model) %models
    for k=1:(M)
        subplot(Nx,Ny,2+i+(k-1)*Ny)
        Nc=size(model{i}.C,1);
        try
        imagesc(reshape(model{i}.C(:,k),12,Nc/12)')
        catch
            imagesc(model{i}.C(:,k))
        end
        colormap(flipud(map))
        caxis([-aC aC])
        if i==1
            ylabel(['C_' num2str(k)])
        end
    end
    subplot(Nx,Ny,2+i+(M)*Ny)
    try
    imagesc(reshape(model{i}.D,12,Nc/12)')
    catch
       imagesc(model{i}.D) 
    end
    colormap(flipud(map))
    caxis([-aC aC])
    if i==1
        ylabel('D')
    end
end
%% Plot R
aR=mean(diag(model{1}.R));
for i=1:length(model) %models
    subplot(Nx,Ny,2+i+(M+1)*Ny)
    imagesc(model{i}.R)
    colormap(flipud(map))
    caxis([-aR aR])
    if i==1
        ylabel('R')
    end
end

%% Plot Q
aQ=2*max(abs(model{1}.Q(:)));
if aQ==0
    aQ=.01*aR./aC^2;
end
for i=1:length(model) %models
    subplot(Nx,Ny,2+i+(M+2)*Ny)
    imagesc(model{i}.Q)
    colormap(flipud(map))
    caxis([-aQ aQ])
    if i==1
        ylabel('Q')
    end
end