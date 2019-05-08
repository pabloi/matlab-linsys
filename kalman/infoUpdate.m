function [new_i,newI,new_x,newP,logL,rejectedSample,oldI]=infoUpdate(CtRinvC,CtRinvY,x,P,rejectZ2threshold,logDetCRC,invCRC,oldI)

  %Parse optional inputs:
  rejectFlag=true;
  if nargin<6 || isempty(rejectZ2threshold)
      rejectFlag=false;
      rejectedSample=false;
  end
  if nargin<7 && (nargout>4 || rejectFlag) %These are needed for fast evaluation of the log-likelihood
      [cholInvCRC,~,invCRC]=pinvchol(CtRinvC);
      logDetCRC=-2*sum(log(diag(cholInvCRC)));
  end
  
  %Do the update:
  if nargin<8 || isempty(oldI)
      [cholOldI,cholOldP,oldI]=pinvchol(P); %Prior information matrix not given, computing from prior covariance
  end
  newI=oldI + CtRinvC;
  new_i=oldI*x + CtRinvY;
  
  if nargout>2 %If new state and covariance were requested:
    [cholP,cholInvP,newP]=pinvchol(newI);
    new_x=newP*new_i; 
    if (nargout>4 || rejectFlag) 
        %[logL,z2]=logLnormal(CtRinvY-CtRinvC*x,CtRinvC*P*CtRinvC+CtRinvC); %This is slow, requires inversion 
        if isempty(cholOldI)
            logDetOldI=0;
        else
            logDetOldI=log(diag(oldI));
        end
        [logL,z2]=logLnormalAlt(CtRinvY-CtRinvC*x,invCRC-newP,sum(log(diag(newP))-logDetOldI)+logDetCRC);
        if rejectFlag && z2>rejectZ2threshold %Reject sample, no update
            rejectedSample=true;  
            newI=oldI;
            return
        end  
    end
  end
end

function [logL,z2]=logLnormalAlt(z,invP,logdetP)
%Faster to evaluate in this case
  halfLog2Pi=0.91893853320467268;
  z2=sum((invP*z).*z,1);
  logL=-.5*(z2) -.5*logdetP-size(z,1)*halfLog2Pi; 
end