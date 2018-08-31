function [x,sP,P]=KFupdatev2(CtRinvY,CtRinvC,x,sP) %(C,R,x,P,y,d,rejectThreshold)
%INPUT: 
%sP is an upper-triangular matrix such that sP'*sP = P (Cholesky decomposition object sP=decomposition(P,'chol'))
%sCRC is the Cholesky decomp of CtRinvC

iP=sP\eye(sP.MatrixSize);%pinv(P,tol); 
iM=iP+CtRinvC; 
M=iM\eye(size(iM));
%M=pinv(iM,tol); 
%K=M*CtRinv; 
%I_KC=M*iP;  %=I -K*C
%x=M*(iP*x+CtRinv*y_d); 
%P=M;%(I_KC)*P; 
x=M*(iP*x+CtRinvY); 
sP=decomposition(M,'chol','upper');
if nargout>2
P=M;
end

end