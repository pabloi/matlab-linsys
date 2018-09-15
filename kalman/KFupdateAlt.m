function [x,P,z]=KFupdateAlt(CtRinvY,CtRinvC,x,P)
%A faster Kalman update step when size(R)>size(P). Requires invertible P.
%To do: is there an equivalent fast version when P is not invertible?

%sP=chol(P); %Since P needs to be invertible, this exists
sP=mycholcov(P);
I=eye(size(P));
isP=I/sP; %This returns the pseudo-inv if sP is under-rank
iP=isP*isP';
iM=iP+CtRinvC; 
isM=chol(iM);
sM=I/isM;
M=sM*sM'; %K=M*CtRinv
%PCM=P*CtRinvC*sM;
%P=P-P*CtRinvC*P + PCM*PCM'; %P=M if P is invertible
%P=P*(I-iP+iP*M*iP);
P=M;

%Do update of state:
%x=P*(iP*x+CtRinvY); 
x=x+P*(CtRinvY-CtRinvC*x); 

if nargout>2
    %If we wanted to check sanity of the update, by evaluating if the
    %innovation (of the state) is within reason given the prior expectations:
    z=z2score(CtRinvY,iM*P*CtRinvC,CtRinvC*x);
end

end