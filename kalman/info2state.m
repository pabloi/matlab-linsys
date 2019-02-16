function [x,P]=info2state(i,I)
  [~,~,P]=pinvchol(I); %Using pseudo-inv
  x=P*i; %This works even if some diagonal elements of I were 0.
  aux=find(diag(I)==0);
  if ~isempty(aux) %Some variances are still Inf
    P(sub2ind(size(P),aux,aux))=Inf; %Restoring infinity
    x(aux)=0;
  end
end
