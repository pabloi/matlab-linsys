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
