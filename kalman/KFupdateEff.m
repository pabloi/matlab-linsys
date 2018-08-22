function [x,iP,P]=KFupdateEff(CtRinvY,CtRinvC,x,iP) %(C,R,x,P,y,d,rejectThreshold)

%tol=1e-8;
%iP=P\eye(size(P));%pinv(P,tol); 
iM=iP+CtRinvC; 
%M=pinv(iM,tol); 
%K=M*CtRinv; 
%I_KC=M*iP;  %=I -K*C
%x=M*(iP*x+CtRinv*y_d); 
%P=M;%(I_KC)*P; 
x=iM\(iP*x+CtRinvY); 
if nargout>2 %Invert P only if necessary
    P=iM\eye(size(iM));%(I_KC)*P; 
end

end