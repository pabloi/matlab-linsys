function Q=robCov(w,prc)
%robCov is a robust covariance matrix estimation of data. Useful if data may contain outliers.
%It uses only an 'inner' percentage of the data (i.e. data lying inside a certain
%ellipsoid) to estimate the covariance matrix. If data comes from a
%multinormal distribution, this estimation is unbiased. There is no
%guarantee for data coming from other distributions (e.g. heavier tailed ones).
%Data is assumed to be 0-mean.
%The procedure is as follows:
%1) Estimate the covariances with the classical MLE estimator.
%2) Use the classical estimator to find the X% of inner-most samples (i.e.
%samples within the 90-th percentile ellipsoid). The other (100-X)% is
%presumed to contain all the outliers.
%3) Estimate the covariance of the reduced samples.
%4) Expand estimate to account for the fact that only the samples closes to
%the origin where used, under the assumption of multivariate normal data.
%INPUT:
%w: MxN matrix, consisting of N-samples of M-dimensional data.
%prc: (optional, default =90%) Percentage of data used for estimation.
%OUTPUT: 
%Q: MxM covariance estimate
%See also: robustCov, estimateParams

%To do: is it worth it to iterate the procedure to converge on the outlier
%samples?

[nD,M]=size(w);
Q=(w*w')/M; %Standard estimate
y=sum(w.*(Q\w),1); %if Q,w weere computed on demeaned data, this is distributed as t^2/M where t^2 ~ Hotelling's T^2 = nD*(M-1)/(M-nD) F_{nD,M-nD}, see https://en.wikipedia.org/wiki/Hotelling%27s_T-squared_distribution
if nargin<2 || isempty(prc)
    prc=90;
elseif prc<1 %Assuming percentile was given in [0,1] range
    prc=round(100*prc);
end
yPRC=prctile(y,prc);
wRob=w(:,y<yPRC);
%First moment of F_{nD,M-nD} = (M-nD)/(M-nD-2)  (~1 if M-nD>>2)
%(approx) Partial first moment to th 90th-percentile:
x=[0:.01:finv(.9,nD,M-nD)];
fp=.01*sum(x.*fpdf(x,nD,M-nD));
k=1/fp; % k is a factor to account for the usage of only the 'first' 90% samples and still get an ~unbiased estimate.
Q=k*(wRob*wRob')/M;

%Some debugging:
%prc=[.99,.95,.9];
%yPRC=prctile(y,prc*100);
%y99=yPRC(1);
%y95=yPRC(2);
%y90=yPRC(3);
% figure;
% plot(w(1,:),w(2,:),'o'); hold on;
% plot(w(1,y>y90),w(2,y>y90),'mo');
% plot(w(1,y>y95),w(2,y>y95),'go');
% plot(w(1,y>y99),w(2,y>y99),'ro');
% th=0:.1:2*pi;
% x=sin(th);
% y=cos(th);
% 
% col={'r','g','m'};
% for j=1:length(prc)
% f99=finv(prc(j),nD,M-nD); %99th percentile of relevant F distribution
% t99=nD*(M-1)/(M-nD) * f99;  %99th of relevant T^2 distribution
% a99=sqrt((t99)./sum([x;y].*(Q\[x;y])));
% plot(a99.*x,a99.*y,col{j})
% end
% %TODO: deal with auto-correlated (ie not white) noise for better estimates.
end