function [x,P,outlierIndx]=KFupdate(C,R,x,P,y,d,rejectThreshold)

if nargin>6 && ~isempty(rejectThreshold)
	[outlierIndx]=detectOutliers(y-d,x,P,C,R);
	if any(~outlierIndx)
	%Update without outliers, by setting outliers to exactly what we expect
	%with inifinite uncertainty
	%y(outlierIndx)=C(outlierIndx,:)*x + d(outlierIndx); %Unnecessary
	R(outlierIndx,:)=1/eps;
	R(:,outlierIndx)=1/eps;

	%Alt: (equivalent) eliminate outliers before update
	%y=y(~outlierIndx);
	%C=C(~outlierIndx,:);
	%R=R(~outlierIndx,~outlierIndx);
	end
end
[x,P]=doUpdate(C,R,x,P,y,d);

end

function [x,P]=doUpdate(C,R,x,P,y,d)
  	%update implements Kalman's update step
  	K=P*C'/(C*P*C'+R);
	x=x+K*(y-C*x-d);
	P=P-K*C*P;
end
