function [transitionMatrix,observationMatrix] = genParamEstim(stateDistrHistory,observationHistory)
[N,D]=size(stateDistrHistory);

%Estimate transition matix p(x_{k+1}|x_k)=p(x_{k+1},x_k)/p(x_k)
%jointDistr=squeeze(mean(stateDistrHistory(2:end,:).*reshape(stateDistrHistory(1:end-1,:),N-1,1,D),1)); %This could be regularized
%Alt implementation: less efficient but does not require large memory
jointDistr=nan(D,D);
for i=1:D
  jointDistr(i,:)=stateDistrHistory(2:N,i)'*stateDistrHistory([1:N-1],:)/(N-1);
end

transitionMatrix=jointDistr./sum(jointDistr,1);

D1=max(observationHistory);
N=length(observationHistory);
%observationDistr=zeros(N,D); %Could be made sparse
%observationDistr(sub2ind([N,D],1:N,observationHistory(:)'))=1;
%jointDistr=squeeze(mean(observationDistr.*reshape(stateDistrHistory,N,1,D),1));
jointDistr=nan(D1,D);
for i=1:D1
    jointDistr(i,:)=sum(stateDistrHistory(observationHistory==i,:)); %This should be regularized somehow
end
observationMatrix=jointDistr./sum(jointDistr,1);
end
