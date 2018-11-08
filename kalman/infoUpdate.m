function [new_i,newI,new_x,newP]=infoUpdate(CtRinvC,CtRinvY,x,P,oldI)
  if nargin<5 || isempty(oldI)
    oldI=pinv(P); %Prior information matrix not given, computing from prior covariance
  end
  newI=oldI + CtRinvC;
  new_i=oldI*x + CtRinvY;
  if nargout>2 %If new state and covariance were requested:
    [~,~,newP]=pinvchol(newI);
    new_x=newP*(oldI*x + CtRinvY);
  end
end
