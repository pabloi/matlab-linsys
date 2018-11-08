function [new_i,newI,new_x,newP,logL,rejectedSample]=infoUpdate(CtRinvC,CtRinvY,x,P,oldI,rejectZ2threshold)
  rejectFlag=true;
  if nargin<6 || isempty(rejectZ2threshold)
      rejectFlag=false;
      rejectedSample=false;
  end
  if nargin<5 || isempty(oldI)
    [~,~,oldI]=pinvchol(P); %Prior information matrix not given, computing from prior covariance
  end
  
  newI=oldI + CtRinvC;
  new_i=oldI*x + CtRinvY;
  
  if nargout>2 %If new state and covariance were requested:
    [~,~,newP]=pinvchol(newI);
    new_x=newP*new_i;
    [logL,z2]=logLnormal(CtRinvY-CtRinvC*x,CtRinvC*P*CtRinvC+CtRinvC); %This is slow, requires inversion
    if rejectFlag && z2>rejectZ2threshold %Reject sample, no update
        rejectedSample=true;  
        newI=oldI;
        return
    end  
  end
end
