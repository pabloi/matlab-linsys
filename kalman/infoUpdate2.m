function [new_i,newI]=infoUpdate2(CtRinvC,CtRinvY,old_i,oldI,rejectZ2threshold,logDetCRC,invCRC)

  %Parse optional inputs:
  %rejectFlag=true;
  %if nargin<6 || isempty(rejectZ2threshold)
  %    rejectFlag=false;
  %    rejectedSample=false;
  %end
  %if nargin<7 && (nargout>4 || rejectFlag) %These are needed for fast evaluation of the log-likelihood
  %    [cholInvCRC,~,invCRC]=pinvchol(CtRinvC);
  %    logDetCRC=-2*sum(log(diag(cholInvCRC)));
  %end
  
  %Do the update:
  %[cholOldI,~,oldI]=pinvchol(P); %Prior information matrix not given, computing from prior covariance
  %cholOldP=chol(P);
  %cholOldI=cholOldP'\eye(size(P));
  %oldI=cholOldI*cholOldI';
  newI=oldI + CtRinvC;
  new_i=old_i + CtRinvY;
  
  if nargout>2 %If new state and covariance were requested:
    %[~,cholInvP,newP]=pinvchol(newI);
    %cholInvP=chol(newI);
    %cholP=cholInvP'\eye(size(P));
    %newP=cholP*cholP';
%     if (nargout>4 || rejectFlag) 
%         %[logL,z2]=logLnormal(CtRinvY-CtRinvC*x,CtRinvC*P*CtRinvC+CtRinvC); %This is slow, requires inversion 
%         logDetOldI=log(diag(oldI));
%         [logL,z2]=logLnormalAlt(CtRinvY-CtRinvC*x,invCRC-newP,sum(log(diag(newI))-logDetOldI)+logDetCRC);
%         if rejectFlag && z2>rejectZ2threshold %Reject sample, no update
%             rejectedSample=true;  
%             newI=oldI;
%             return
%         end  
%     end
  end
end

function [logL,z2]=logLnormalAlt(z,invP,logdetP)
%Faster to evaluate in this case
  halfLog2Pi=0.91893853320467268;
  z2=sum((invP*z).*z,1);
  logL=-.5*(z2) -.5*logdetP-size(z,1)*halfLog2Pi; 
end