function [previ,prevI]=state2info(x0,prevP)

  %Sanitize init Info matrix if P0 contains infinite elements
  infVariances=isinf(diag(prevP));
  if any(infVariances)
      %Define info matrix from cov matrix:
      prevI=zeros(size(prevP));
      aux=inv(prevP(~infVariances,~infVariances)); %This inverse needs to exist, no such thing as absolute certainty
      prevI(~infVariances,~infVariances)=aux; %Computing inverse of the finite submatrix of P0
      x0(infVariances)=0; %No estimation available
  else
      prevI=inv(prevP); %This inverse needs to exist, no such thing as absolute certainty
  end
  previ=prevI*x0;
