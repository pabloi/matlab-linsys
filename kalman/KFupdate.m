function [x,P]=KFupdate(CtRinvY,CtRinvC,x,P)

%tol=1e-8;
sP=chol(P); %To ensure symmetry
isP=sP\eye(size(sP));
iP=isP*isP';
%iP=P\eye(size(P)); %For some reason, this is much faster than pinv
iM=iP+CtRinvC; 
%P=iM\eye(size(iM));%Same as above
isM=chol(iM);
sM=isM\eye(size(isM));
P=sM*sM';

%If we wanted to check sanity of the update, by evaluating if the
%innovation (of the state) is within reason given the prior expectations:
%z=z2score(CtRinvY,iM*P*CtRinvC,CtRinvC*x);

%M=pinv(iM,tol); 
%K=M*CtRinv; 
%I_KC=M*iP;  %=I -K*C
%x=M*(iP*x+CtRinv*y_d); 
%P=M;%(I_KC)*P; 

%Do update of state:
x=P*(iP*x+CtRinvY); 


end