function [x,P]=KFupdateEff(CtRinv,CtRinvC,x,P,y_d)
    tol=1e-8;
    iP=pinv(P,tol);
    iM=iP+CtRinvC;
    %M=pinv(iM,tol);
    %K=M*CtRinv;
    %KC=M*CtRinvC;
    K=iM\CtRinv;
    %KC=iM\CtRinvC;
    %I_KC=(eye(size(P))-KC);
    I_KC=iM\iP;
    P=I_KC*P; %Should equal pinv(iM)
    x=I_KC*x+K*(y_d); 
end