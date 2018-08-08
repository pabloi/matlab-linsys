function [A,B,C,D,Q,R,x0,P0]=estimateParams(Y,U,X,P,Pt)
%M-step of EM estimation for LTI-SSM
%INPUT:
%Y = output of the system, D2 x N
%U = input of the system, D3 x N
%X = state estimates of the system (Kalman-smoothed), D1 x N
%P = covariance of states (Kalman-smoothed), D1 x D1 x N
%Pp = covariance of state transitions (Kalman-smoothed), D1 x D1 x (N-1),
%evaluated at k+1|k
%See Cheng and Sabes 2006, Ghahramani and Hinton 1996, Shumway and Stoffer 1982

%This is equivalent to the true M-step if no uncertainty in X (P=Pt=0)
% [A,B,Q] = estimateAB(X, U);
% [C,D,R] = estimateCD(Y,X, U);
% x0=X(:,1);
% P0=[];
% return

%True M-step
%tol=1e-8;

%Define vars:
SP1=sum(P(:,:,1:end-1),3);
SP2=sum(P(:,:,2:end),3);
SP=sum(P,3);
SPt=sum(Pt,3);
[D1,N]=size(X);

%x0,P0:
x0=X(:,1);
P0=P(:,:,1);

%A,B:
xu=X(:,1:end-1)*U(:,1:end-1)';
uu=U(:,1:end-1)*U(:,1:end-1)';
xu1=X(:,2:end)*U(:,1:end-1)';
xx=X(:,1:end-1)*X(:,1:end-1)';
xx1=X(:,2:end)*X(:,1:end-1)';
O=[SP1+xx xu; xu' uu];
%AB=[sum(Pt,3)+xx1 xu1]*pinv(O,1e-8);
AB=[SPt+xx1 xu1]/O; %More efficient than above
%AB=lsqminnorm(O,[SPt+xx1 xu1]',tol)'; %More stable than above
%Notice that in absence of uncertainty in states, this reduces to
%[A,B]=X+/[X;U], where X+ is X one step in the future
A=AB(:,1:D1);
B=AB(:,D1+1:end);

%C,D:
xu=X*U';
uu=U*U';
xx=X*X';
O=[SP+xx xu; xu' uu];
%CD=[Y*X' Y*U']*pinv(O,tol);
%CD=lsqminnorm(O,[X;U]*Y',tol)'; %More efficient than line above
CD=Y*[X; U]'/O; %Notice that in absence of uncertainty in states, this reduces to [C,D]=Y/[X;U]
C=CD(:,1:D1);
D=CD(:,D1+1:end);

%Q,R: 
%Adaptation of Shumway and Stoffer 1982: (there B=D=0 and C is fixed), but
%consistent with Ghahramani and Hinton 1996, and Cheng and Sabes 2006
z=Y-C*X-D*U;
w=X(:,2:N)-A*X(:,1:N-1)-B*U(:,1:N-1);

% MLE estimator of Q, under the given assumptions:
Q2=(SP2-2*A*SPt'+A*SP1*A')/(N-1);
Q=(w*w')/(N-1)+Q2; %Not designed to deal with outliers, autocorrelated w
%Robust covariance estimation (this deals with outliers, not autocorr):
%Q = robustcov(w') +Q2;
%Cheap variant of robust estimation:
Q=robCov(w) +Q2;

%Estimate R:
R=(z*z'+C*SP*C')/N;

%A variant in the estimation of P0, to not make it monotonically decreasing
%as number of iterations increase:
%iP=pinv(P0,1e-8);
%iP0=iP+C'*(R\C);
%P0=Q+A*(iP0\A'); %P0=Q+A*P0*A can be a proxy
P0=Q+A*P0*A';

%%Expression of covariances should be symmetric and PSD, but may not be because of numerical issues:
P0=positivize(P0);
Q=positivize(Q);
%R=positivize(R);
end

function Q=robCov(w)
%Note: if we dont have exact extimates of A,B, then the residuals w are not
%iid gaussian. They will be autocorrelated AND have outliers with respect
%to the best-fitting multivariate normal. Thus, we benefit from doing a
%more robust estimate, especially to avoid local minima in trueEM
[nD,M]=size(w);
Q=(w*w')/M; %Standard estimate
y=sum(w.*(Q\w),1); %if Q,w where computed on demeaned data, this is distributed as t^2/M where t^2 ~ Hotelling's T^2 = nD*(M-1)/(M-nD) F_{nD,M-nD}, see https://en.wikipedia.org/wiki/Hotelling%27s_T-squared_distribution
yPRC=prctile(y,90);
wRob=w(:,y<yPRC);
%First moment of F_{nD,M-nD} = (M-nD)/(M-nD-2)  (~1 if M-nD>>2)
%(approx) Partial first moment to th 90th-percentile:
x=[0:.01:finv(.9,nD,M-nD)];
fp=.01*sum(x.*fpdf(x,nD,M-nD));
k=1/fp;
Q=k*(wRob*wRob')/M;% k is a factor to account for the usage of only the first 90% samples and still get an ~unbiased estimate.

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