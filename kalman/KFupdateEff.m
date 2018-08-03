function [x,Pinv]=KFupdateEff(CtRinv,CtRinvC,x,Pinv,y_d)
    tol=1e-8;
    %iP=pinv(P,tol);
    iM=Pinv+CtRinvC;
    %M=pinv(iM,tol);
    %K=M*CtRinv;
    %%KC=M*CtRinvC;
    %I_KC=M*iP;
    %Alt:
    %K=iM\CtRinv;
    %%KC=iM\CtRinvC;
    %%I_KC=(eye(size(P))-KC);
    %I_KC=M*iP;
    x=iM\(Pinv*x + CtRinv*y_d); 
    Pinv=iM;
    %P=eye(size(iM))/iM;%I_KC*P;
    
end