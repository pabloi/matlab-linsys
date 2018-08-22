function [x,P]=KFupdate(CtRinvY,CtRinvC,x,P)

%tol=1e-8;
iP=P\eye(size(P));%For some reason, this is much faster than pinv
iM=iP+CtRinvC; 
P=iM\eye(size(iM));%Same as above
%M=pinv(iM,tol); 
%K=M*CtRinv; 
%I_KC=M*iP;  %=I -K*C
%x=M*(iP*x+CtRinv*y_d); 
%P=M;%(I_KC)*P; 
x=P*(iP*x+CtRinvY); 

end