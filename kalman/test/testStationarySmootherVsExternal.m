addpath(genpath('../../'))
%% Create model:
D1=5;
D2=200;%100;
%CS 2006 gets progressively slower for larger D2 (linear execution time with D2 for large D2).
%This implementation grows linearly too but with the SMALLEST of D1,D2. For
%small D2, CS2006 is slightly faster, as it does not enforce covariance
%matrices to be PSD. This sometimes results in ugly filtering (especially
%with large covariance matrices, the smoothing does not work well, even
%being less accurate than this implementation's filtering).
A=diag(rand(D1,1));
A=.9999*A; %Setting the max eigenvalue to .9999
%B=3*randn(D1,1);
%B=B./sign(B); %Forcing all elements of B to be >0, WLOG
B=(eye(size(A))-A)*ones(size(A,1),1); %WLOG, arbitrary scaling so all states asymptote at 1
N=2000;
U=[zeros(300,1);ones(N,1);zeros(N/2,1)]'; %Step input and then removed
C=randn(D2,D1);
D=randn(D2,1);
Q=eye(D1)*1e-1;
Q=zeros(D1);
R=1*eye(D2); %CS2006 performance degrades (larger state estimation errors) for very small R

%% Simulate
NN=size(U,2);
x0=zeros(D1,1);
[Y,X]=fwdSim(U,A,B,C,D,x0,Q,R);
[Y1,X1]=fwdSim(U,A,B,C,D,x0,[],[]); %Noiseless simulation, for comparison
%Y(:,400)=NaN;
X=X(:,1:end-1);

%% Define initial uncertainty
P0=zeros(D1);
P0=diag(Inf*ones(D1,1));
P0=1e1*eye(D1);

%% Do kalman smoothing with true params, no fast mode, no reduction
tic
mdl=1;
opts.noReduceFlag=true;
opts.fastFlag=0;
[Xs,Ps,Pt,Xf,Pf,Xp,Pp,~,logL(mdl)]=statKalmanSmoother(Y,A,C,Q,R,x0,P0,B,D,U,opts); %Kalman smoother estimation of states, given the true parameters (this is the best possible estimation of states)
Xcs=Xs;
res(mdl)=norm(Xs-X,'fro')^2;
maxRes(mdl)=max(sum((Xs-X).^2));
name{mdl}='KS';
res2KS(mdl)=norm(Xs-Xcs,'fro');
tc(mdl)=toc;

%% Do kalman smoothing with true params, no fast mode, reduction
tic
mdl=2;
opts.noReduceFlag=false;
opts.fastFlag=0;
[Xs,Ps,Pt,Xf,Pf,Xp,Pp,~,logL(mdl)]=statKalmanSmoother(Y,A,C,Q,R,x0,P0,B,D,U,opts); %Kalman smoother estimation of states, given the true parameters (this is the best possible estimation of states)
res(mdl)=norm(Xs-X,'fro')^2;
maxRes(mdl)=max(sum((Xs-X).^2));
name{mdl}='KS red';
res2KS(mdl)=norm(Xs-Xcs,'fro');
tc(mdl)=toc;

%% Do kalman smoothing with true params, fast mode, reduction
tic
mdl=3;
opts.noReduceFlag=false;
opts.fastFlag=1;
[Xs,Ps,Pt,Xf,Pf,Xp,Pp,~,logL(mdl)]=statKalmanSmoother(Y,A,C,Q,R,x0,P0,B,D,U,opts); %Kalman smoother estimation of states, given the true parameters (this is the best possible estimation of states)
res(mdl)=norm(Xs-X,'fro')^2;
maxRes(mdl)=max(sum((Xs-X).^2));
name{mdl}='KSred fast';
res2KS(mdl)=norm(Xs-Xcs,'fro');
tc(mdl)=toc;

%% Use Info smoother:
mdl=4;
tic;
[is,Is,iif,If,ip,Ip,Xs,Ps,~]=statInfoSmoother2(Y,A,C,Q,R,x0,P0,B,D,U,opts);
res(mdl)=norm(Xs-X,'fro')^2;
maxRes(mdl)=max(sum((Xs-X).^2));
logL(mdl)=nan;
name{mdl}='IS';
res2KS(mdl)=norm(Xs-Xcs,'fro');
tc(mdl)=toc;

%% Use Cheng & Sabes code
[folder]=fileparts(mfilename('fullpath'));
addpath(genpath([folder '/../../ext/lds-1.1/']))

%%(matlab version), reduced:
mdl=5;
opts.noReduceFlag=false;
tic
[Xs,Ps,Pt,logL(mdl)]=statKalmanSmootherCS2006Matlab(Y,A,C,Q,R,x0,P0,B,D,U,opts);
res(mdl)=norm(Xs-X,'fro')^2;
maxRes(mdl)=max(sum((Xs-X).^2));
name{mdl}='CSred';
res2KS(mdl)=norm(Xs-Xcs,'fro');
tc(mdl)=toc;

%% C version, reduced:
mdl=6;
opts.noReduceFlag=false;
tic
[Xs,Ps,Pt,logL(mdl)]=statKalmanSmootherCS2006(Y,A,C,Q,R,x0,P0,B,D,U,opts);
res(mdl)=norm(Xs-X,'fro')^2;
maxRes(mdl)=max(sum((Xs-X).^2));
name{mdl}='CSred C';
res2KS(mdl)=norm(Xs-Xcs,'fro');
tc(mdl)=toc;

%% CS, Matlab version, no reduce (C version is terribly slow for large problems)
mdl=7;
opts.noReduceFlag=true;
tic
[Xs,Ps,Pt,logL(mdl)]=statKalmanSmootherCS2006Matlab(Y,A,C,Q,R,x0,P0,B,D,U,opts);
res(mdl)=norm(Xs-X,'fro')^2;
maxRes(mdl)=max(sum((Xs-X).^2));
name{mdl}='CS';
res2KS(mdl)=norm(Xs-Xcs,'fro');
tc(mdl)=toc;


%% Visualize results
fh=figure;
subplot(5,1,1) %Bar of residuals
bar(sqrt(res),'EdgeColor','none')
set(gca,'XTickLabel',name)
title('RMSE residuals')
axis tight
aa=axis;
axis([aa(1:2) 0 sqrt(max(res))*1.02])
grid on

subplot(5,1,2) %Bar of residuals to KS
bar(res2KS,'EdgeColor','none')
set(gca,'XTickLabel',name)
title('RMSE residuals to KS')
axis tight
aa=axis;
axis([aa(1:2) 0 1e-12])
grid on

subplot(5,1,3) %Bar of max res
bar(sqrt(maxRes),'EdgeColor','none')
set(gca,'XTickLabel',name)
title('Max sample rMSE')
axis tight
aa=axis;
axis([aa(1:2) 0 sqrt(max(maxRes))*1.1])
grid on

subplot(5,1,4) %Bar of logL
bar(logL,'EdgeColor','none')
set(gca,'XTickLabel',name)
title('LogL')
axis tight
aa=axis;
axis([aa(1:2) logL(1)+[-1 1]*1e-5])
grid on

subplot(5,1,5) %Bar of running times
bar(tc/tc(3),'EdgeColor','none')
set(gca,'XTickLabel',name,'YScale','log','YTick',[.5 1 1e1 1e2 1e3])
axis tight
aa=axis;
axis([aa(1:2) .5 2e2])
title(['Relative running time, KSred fast=' num2str(tc(3))])
grid on

%Save fig
print(fh,['results.eps'],'-depsc','-tiff','-r2400')
save results.mat res tc logL maxRes res2KS
