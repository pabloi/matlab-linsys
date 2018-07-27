function [x,P,outlierIndx]=KFupdate(C,R,x,P,y,d,rejectThreshold)

if nargin>6
	[outlierIndx]=detectOutliers(y-d,x,P,C,R,rejectThreshold);
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
[x,P]=doUpdate(C,R,x,P,y,d);

end

function [x,P]=doUpdate(C,R,x,P,y,d)
  	%update implements Kalman's update step
    %Fast and stable-ish implementation:
    CP=C*P;
    S=CP*C'+R;
    %K=P*C'*pinv(S,1e-5); %This is equivalent to K=lsqminnorm(C*P,S,1e-5)'
    K=lsqminnorm(S,CP,1e-5)';
    AA=(eye(size(P))-K*C);
    P=AA*P;
    x=x+K*(y-C*x-d);          
end