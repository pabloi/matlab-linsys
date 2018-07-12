function [outlierIndx]=detectOutliers(y,x,P,C,R)
m=size(y,1);
th=9; %Threshold, limit value for (x-mu)'*sigma^{-1}*(x-mu)/m
expY=C*x;
Py=C*P*C' + R;
innov=y-expY;
%logp=innov'*pinv(Py)*innov; %-1/2 of log(p) of observation
auxTh=(1+(th-1)/sqrt(m))*m; %Formally I should use the inverse of chi-square tail as we can think of auxLogP as the sum of m normally distributed variables
auxLogP=pinv(Py)*innov;
%innov.*auxLogP
auxDist=innov.*auxLogP;
outlierSamples=sum(auxDist)>auxTh; %Values of 1 indicate likely outliers
outlierIndx=false(size(auxDist));
outlierIndx(:,outlierSamples)=true;
outlierIndx=auxDist> sqrt(2)*erfcinv(erfc(sqrt(th/2))/m) %Bonferroni-corrected probability of tail. For m=1 this means more than sqrt(th) std away

%if any(outlierIndx)
%    disp(['Found outliers:' num2str(find(outlierIndx)')])
%    y
%end

%ALT: do it recursively by finding the lowest log(p), if it passes the threshold
%exclude it, then re-compute log(p) without considering it, find the second lowest
%value, and so forth.

end
