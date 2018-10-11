function [fh] = vizModels(model)

M=max(cellfun(@(x) size(x.J,1),model));
fh=figure('Units','Normalized','OuterPosition',[0 0 1 1]);
Ny=2+length(model);
Md=size(model{1}.D,2);
Nx=M+Md+2;

%% Compute output and residuals
U=[zeros(Md,100) ones(Md,1000)];
for i=1:length(model)
    [model{i}.J,model{i}.B,model{i}.C,~,~,model{i}.Q] = canonizev4(model{i}.J,model{i}.B,model{i}.C,[],model{i}.Q);
    [Y2,X2]=fwdSim(U,model{i}.J,model{i}.B,model{i}.C,model{i}.D,[],[],[]);
    model{i}.smoothStates=X2;
    model{i}.smoothOut=Y2;
end
%% Plot STATES
clear p
for k=1:length(model)
    %if k==1
        plotInd=1:size(model{k}.J,1);
    %else
    %    thisTau=-1./log(eig(model{k}.J));
    %    [plotInd,~]=bestPairedMatch(thisTau,lastTau);
    %end
    %lastTau=-1./log(eig(model{k}.J));
    for i=1:size(model{k}.J,1)
    subplot(Nx,Ny,Ny*(plotInd(i)-1)+[1:2]) %States
    hold on
    set(gca,'ColorOrderIndex',k)
    p{i}(k)=plot(model{k}.smoothStates(i,:),'LineWidth',2,'DisplayName',[model{k}.name ', \tau=' num2str(-1./log(model{k}.J(i,i)),3)]);
    if k==1
        ylabel(['State ' num2str(i)])
        title('(Smoothed) Step-response states')
    end
    end
if k==length(model)
    for i=1:M
         subplot(Nx,Ny,Ny*(i-1)+[1:2]) %
        legend(findobj(gca,'Type','Line'),'Location','SouthEast')
    end
end
end

%% Define colormap:
ex1=[1,0,0];
ex2=[0,0,1];
mid=ones(1,3);
N=100;
map=[ex1.*[N:-1:1]'/N + mid.*[0:N-1]'/N; mid; ex2.*[0:N-1]'/N + mid.*[N:-1:1]'/N];
%% Plot C and D columns
%aC=max(abs(model{1}.C(:)));
aC=.5*max(cellfun(@(x) max(abs(x.C(:))),model));
for i=1:length(model) %models
    plotInd=1:size(model{i}.J,1);
    for k=1:size(model{i}.J,1)
        subplot(Nx,Ny,(plotInd(k)-1)*Ny+2+i)
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

    for k=1:Md
        subplot(Nx,Ny,2+i+(M+k-1)*Ny)
    try
    imagesc(reshape(model{i}.D(:,k),12,Nc/12)')
    catch
       imagesc(model{i}.D(:,k))
    end
    colormap(flipud(map))
    caxis([-aC aC])
    if i==1
        ylabel(['D_' num2str(k)])
    end
    end
end
%% Plot R
aR=mean(diag(model{1}.R));
for i=1:length(model) %models
    subplot(Nx,Ny,2+i+(M+Md)*Ny)
    imagesc(model{i}.R)
    colormap(flipud(map))
    caxis([-aR aR])
    if i==1
        ylabel('R')
    end
end

%% Plot Q
aQ=max(cellfun(@(x) max(abs(x.Q(:))),model));
if aQ==0
    aQ=.01*aR./aC^2;
end
for i=1:length(model) %models
    subplot(Nx,Ny,2+i+(M+Md)*Ny)
    imagesc(model{i}.Q)
    colormap(flipud(map))
    caxis([-aQ aQ])
    if i==1
        ylabel('Q')
    end
end
