function [fh] = vizModels(model)

M=max(cellfun(@(x) size(x.J,1),model));
fh=figure('Units','Normalized','OuterPosition',[0 0 1 1]);
Ny=2+length(model);
Md=size(model{1}.D,2);
Nx=M+Md+2;
%% Compute output and residuals
U=[zeros(Md,100) ones(Md,900)];
for i=1:length(model)
    %[model{i}.J,model{i}.B,model{i}.C,~,~,model{i}.Q] = canonize(model{i}.J,model{i}.B,model{i}.C,[],model{i}.Q,[],canon,900);
    [Y2,X2]=fwdSim(U,model{i}.J,model{i}.B,model{i}.C,model{i}.D,[],[],[]);
    model{i}.smoothStates=X2;
    model{i}.smoothOut=Y2;
    rmfield(model{i},'A'); %Just in case
end
%% Plot STATES
clear p
%To find the largest model first:
modelOrders=cellfun(@(x) size(x.J,1),model);
largestModel=find(modelOrders==max(modelOrders),1);
%largestMdlTau=-1./log(diag(model{largestModel}.J)-[.1*diag(model{largestModel}.J,1);0]); %This makes complex poles indistinguishable
allTaus=cellfun(@(x) -1./log(diag(x.J)),model,'UniformOutput',false);
%sortedTau=sortTau(largestMdlTau,allTaus);
allC=cellfun(@(x) x.C,model,'UniformOutput',false);
refC=model{largestModel}.C;
sortedTau=sortC(refC,allC);
for k=1:length(model)
    complexTau=allTaus{k};
    for i=1:modelOrders(k)
        subplot(Nx,Ny,Ny*(sortedTau{k}(i)-1)+[1:2]) %States
        hold on
        set(gca,'ColorOrderIndex',k)
            %dispName=[model{k}.name ', \tau=' num2str(complexTau(i),3)];
            dispName=[model{k}.name];
        p{i}(k)=plot(model{k}.smoothStates(i,:),'LineWidth',2,'DisplayName',dispName);
        if k==1
            ylabel(['State ' num2str(i)])
            title('(Smoothed) Step-response states')
        end
    end
    if k==length(model) %Add legends
        for i=1:M
             subplot(Nx,Ny,Ny*(i-1)+[1:2]) %
            legend(findobj(gca,'Type','Line'),'Location','SouthEast')
        end
    end
end

%% Add plot of poles/time-constants
subplot(Nx,Ny,Ny*(largestModel)+[1]) %States
allTaus=cellfun(@(x) -1./log(eig(x.J)),model,'UniformOutput',false);
hold on
for k=1:length(model)
   scatter(k*ones(size(allTaus{k}))+.2*sign(imag(allTaus{k})),real(allTaus{k}),'filled') 
end
set(gca,'YScale','log','YLim',[2 5e3],'XLim',[-5 5+length(model)],'XTick',[],'YTick',[1e1 1e2 1e3],'YTickLabel',{'1e1','1e2','1e3'})
grid on
%% Define colormap:
ex1=[1,0,0];
ex2=[0,0,1];
mid=ones(1,3);
N=100;
map=[ex1.*[N:-1:1]'/N + mid.*[0:N-1]'/N; mid; ex2.*[0:N-1]'/N + mid.*[N:-1:1]'/N];
%% Plot C and D columns
%aC=max(abs(model{1}.C(:)));
mdl=cellfun(@(x) linsys.struct2linsys(x),model,'UniformOutput',false);
SNR=cellfun(@(x) x.SNR(1000),mdl,'UniformOutput',false);
%SNR=cellfun(@(x) SNR(linsys.struct2linsys(x),1000),model,'UniformOutput',false);
aC=.5*max(cellfun(@(x) max(abs(x.C(:))),model));
for i=1:length(model) %models
    mdlOrder=size(model{i}.J,1);
    %plotInd=1:mdlOrder; %Plot in ascending order of time constants using the first slots
    %ALT: find the closest time-constant in the largest order model
    for k=1:mdlOrder
        subplot(Nx,Ny,(sortedTau{i}(k)-1)*Ny+2+i)
        Nc=size(model{i}.C,1);
        try
            imagesc(reshape(model{i}.C(:,k),12,Nc/12)')
        catch
            imagesc(model{i}.C(:,k))
        end
        colormap(flipud(map))
        caxis([-aC aC])
        set(gca,'XTick',[],'YTick',[])
        %title(['max b/\sqrt{q}=' num2str(max(model{i}.B(k,:))/sqrt(model{i}.Q(k,k)),2)]);% ', max q_ij=' num2str(max(model{i}.Q(k,:)))])
        %title(['b^2/q=' num2str(max(model{i}.B(k,:)).^2/(model{i}.Q(k,k)),3)]);% ', max q_ij=' num2str(max(model{i}.Q(k,:)))])
        %title(['x^2/q=' num2str(max(model{i}.smoothStates(k,:)).^2/(model{i}.Q(k,k)),2)]);% ', max q_ij=' num2str(max(model{i}.Q(k,:)))])
        title(['SNR(1000)=' num2str(SNR{i}(k))])
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
end %Main function

function sortedTau=sortTau(refTau,allTaus)
    %Sorting heuristic to get the best possible match accross tau values
    sortedTau=cell(size(allTaus));
    for i=1:length(allTaus) %Each model to sort
        tauCopy=real(refTau);
        thisTau=allTaus{i};
        sortedTau{i}=nan(size(thisTau));
        for k=1:length(thisTau)
            tau=thisTau(k);
            tauDist=abs(tau./tauCopy -1);
            sortedTau{i}(k)=find(tauDist==nanmin(tauDist),1,'first');
            tauCopy(sortedTau{i}(k))=0; %So it doesn't get re-selected
        end
    end
end

function sortedTau=sortC(refC,allC)
    %Sorting heuristic to get the best possible match accross tau values
    sortedTau=cell(size(allC));
    for i=1:length(allC) %Each model to sort
        tauCopy=real(refC);
        thisTau=allC{i};
        sortedTau{i}=nan(1,size(thisTau,2));
        allDist=1-abs(thisTau'*tauCopy)./sqrt(sum(thisTau.^2,1)'.*sum(tauCopy.^2,1));
        for k=1:size(thisTau,2) %Assign the best pair, one pair at a time
            [ii,jj]=find(allDist==nanmin(nanmin(allDist)),1,'first');
            sortedTau{i}(ii)=jj;
            allDist(ii,:)=NaN;
            allDist(:,jj)=NaN; %Neither can get re-sleected
        end
    end
end