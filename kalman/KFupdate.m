function [x,P,outlierIndx]=KFupdate(C,R,x,P,y,d,rejectThreshold)

if nargin>6
	[outlierIndx]=detectOutliers(y-d,x,P,C,R,rejectThreshold);
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
    %Fast and stable-ish implementation:
    if (sum(C.^2)*trace(P))>100*trace(R) %Avoid ill-conditioned situations where computing K would be too noisy
        K=pinv(C);
        x=K*(y-d);
        P=K*R*K'; %I think this should be the case
        %P=.01*P;
        %Need to work out exact update when CPC' >> R and R>> CPC'
    else
        K=P*C'/(C*P*C'+R);
        AA=(eye(size(P))-K*C);
        P=AA*P;
        x=x+K*(y-C*x-d);    
    end
    %Very slow, but (in theory) stable implementation:
    %S=C*P*C'+R;
    %K=P*C'*pinv(S);
    %Supposedly faster, but very unstable:
    %K=P*(C'/S); 

    %CRC=C'*(R\C);
    %Pinv=pinv(P);
    %KC=(Pinv+CRC)\CRC;
    %K=(Pinv+CRC)\(C'/R);
    %x=x+K*(y-d) +KC*x;
    %P=P-KC*P;
end

function [x,P]=doUpdateEff(C,Rinv,x,P,y,d)
  %update implements Kalman's update step
%Same as doUpdate but taking R^{-1}, as input instead of R and using the
%Sherman-Morrison-Woodbury Lemma for matrix inversion, which allows us to
%compute a cheaper inverse when the size of R (output space) is much
%larger than the size of P (latent space)
%In practice: works like shit. I dont know why.
A=Rinv*C;
B=C'*A;
%TODO: for stationary filters, we can cache the values of A,B for even faster performance
D=pinv(P)+B;
CtSinv=(eye(size(A,2))-B*pinv(D))*A';
K=P*CtSinv;
KC=K*C;
x=K*(y-d)+(I-KC)*x;
P=(I-KC)*P;
end