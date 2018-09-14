function [x,P,z]=KFupdateAlt(CtRinvY,CtRinvC,x,P)
%A faster Kalman update step when size(R)>size(P). Requires invertible P.
%To do: is there an equivalent fast version when P is not invertible?

sP=chol(P); %Since P needs to be invertible, this exists
isP=sP\eye(size(sP));
iP=isP*isP';
iM=iP+CtRinvC; 
isM=chol(iM);
sM=isM\eye(size(isM));
P=sM*sM';

%Do update of state:
x=P*(iP*x+CtRinvY); 

if nargout>3
    %If we wanted to check sanity of the update, by evaluating if the
    %innovation (of the state) is within reason given the prior expectations:
    z=z2score(CtRinvY,iM*P*CtRinvC,CtRinvC*x);
end

end