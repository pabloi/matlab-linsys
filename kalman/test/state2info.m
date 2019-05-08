function [previ,prevI]=state2info(x0,prevP)

  %Sanitize init Info matrix if P0 contains infinite elements
  dP=diag(prevP);
  infVariances=isinf(dP);
  if any(infVariances)
      %Define info matrix from cov matrix:
      prevI=zeros(size(prevP));
      aux=~infVariances;
      redP=prevP(aux,aux); %Selecting submatrix of components that have finite  variance
      [~,~,redI]=pinvchol2(redP);
      prevI(aux,aux)=redI; %Computing inverse of the finite submatrix of P0
      x0(infVariances)=0; %No estimation available
  else %No infinite variances, simply compute pseudo-inverse
      [~,~,prevI]=pinvchol(prevP);
  end
  previ=prevI*x0;
  aux=find(dP==0); %If some elements of prior were infinity
  prevI(aux,:)=0;
  prevI(:,aux)=0;
  prevI(sub2ind(size(prevI),aux,aux))=Inf; %Dealing Inf to diagonals
