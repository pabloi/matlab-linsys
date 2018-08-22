function [x,P]=KFupdate(CtRinv,CtRinvC,x,P,y_d) %(C,R,x,P,y,d,rejectThreshold)

	[outlierIndx]=detectOutliers(y_d,x,P,C,R,rejectThreshold);
	if any(~outlierIndx)
	%Update without outliers, by setting outliers to exactly what we expect
	%with inifinite uncertainty
	R(outlierIndx,:)=1/eps;
	R(:,outlierIndx)=1/eps;

	%Alt: (equivalent) eliminate outliers before update
	%y=y(~outlierIndx);
	%C=C(~outlierIndx,:);
	%R=R(~outlierIndx,~outlierIndx);
	end
end
tol=1e-8;
iP=P\eye(size(P));%pinv(P,tol); 
iM=iP+CtRinvC; 
%M=pinv(iM,tol); 
%K=M*CtRinv; 
%I_KC=M*iP;  %=I -K*C
%x=M*(iP*x+CtRinv*y_d); 
%P=M;%(I_KC)*P; 
x=iM\(iP*x+CtRinv*y_d); 
P=iM\eye(size(iM));%(I_KC)*P; 

end