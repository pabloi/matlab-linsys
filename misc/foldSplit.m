function [foldedData] = foldSplit(data,Nfolds)
%Does folding of data for cross-validation. 
%The folding is performed by taking 1 every other Nfolds points. 
%For example, for 2-fold, the odd datapoints will end up in fold 1, while
%the even ones will end up in fold 2. Datapoints not used are not removed, 
%but replaced with NaNs, so that each fold has the same size as the
%original data matrix. In this way it can be used directly with sysId
%methods such as EM, and parameters obtained across different folds and
%full data can be compared to one another. Folding is performed along the
%first dimension.
foldedData=cell(Nfolds,1);
for i=1:Nfolds
   foldedData{i}=nan(size(data));
   foldedData{i}(i:Nfolds:end,:)=data(i:Nfolds:end,:); 
end

