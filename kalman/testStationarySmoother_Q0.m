%% Create model:
D1=2;
D2=100;
N=1000;
A=randn(D1);
A=.98*A./max(abs(eig(A))); %Setting the max eigenvalue to .98
A=[.95,0;0,.99];
A=jordan(A); %Using A in its jordan canonical form so we can compare identified systems, WLOG
%B=3*randn(D1,1);
%B=B./sign(B); %Forcing all elements of B to be >0, WLOG
B=(eye(size(A))-A)*ones(size(A,1),1); %WLOG, arbitrary scaling
U=[zeros(500,1);ones(N,1);zeros(N/2,1)]'; %Step input and then removed
C=randn(D2,D1);
D=randn(D2,1);
Q=zeros(D1);
R=eye(D2)*.01;

%% Simulate
addpath(genpath('../aux/'))
NN=size(U,2);
x0=zeros(D1,1);
[Y,X]=fwdSim(U,A,B,C,D,x0,Q,R);
[Y1,X1]=fwdSim(U,A,B,C,D,x0,[],[]); %Noiseless simulation, for comparison

%% Do kalman smoothing with true params
tic
fastFlag=0;
[Xsf,Psf,~,Xff,Pff]=statKalmanSmoother(Y,A,C,Q,R,[],[],B,D,U,false,fastFlag); %Kalman smoother estimation of states, given the true parameters (this is the best possible estimation of states)
tf=toc;
tic
fastFlag=[];
[Xs,Ps,Pt,Xf,Pf,Xp,Pp]=statKalmanSmoother(Y,A,C,Q,R,[],[],B,D,U,false,fastFlag); %Kalman smoother estimation of states, given the true parameters (this is the best possible estimation of states)
ts=toc;

%% Visualize results
figure
for i=1:2
    subplot(3,1,i)
    plot(Xs(i,:),'DisplayName','Smoothed')
    hold on
    plot(Xf(i,:),'DisplayName','Filtered')
    plot(Xsf(i,:),'DisplayName','Fast Smooth')
    plot(Xff(i,:),'DisplayName','Fast Filt')
    plot(X(i,:),'DisplayName','Actual')
    
    legend
    if i==1
        title(['Fast runtime= ' num2str(tf) ', Regular runtime= ' num2str(ts)]);
    end

end
subplot(3,1,3)
for i=1:2
     hold on
     set(gca,'ColorOrderIndex',1)
    plot(Xs(i,:)-X(i,1:end-1),'DisplayName','Smoothed')
    plot(Xf(i,:)-X(i,1:end-1),'DisplayName','Filtered')
    plot(Xsf(i,:)-X(i,1:end-1),'DisplayName','FastSmooth')
    plot(Xff(i,:)-X(i,1:end-1),'DisplayName','FastFilt')
        set(gca,'ColorOrderIndex',1)
    bar(1500+500*i,sqrt(mean((X(i,1:end-1)-Xs(i,:)).^2)),'BarWidth',100,'EdgeColor','None')
    bar(1600+500*i,sqrt(mean((X(i,1:end-1)-Xf(i,:)).^2)),'BarWidth',100,'EdgeColor','None')
    bar(1700+500*i,sqrt(mean((X(i,1:end-1)-Xsf(i,:)).^2)),'BarWidth',100,'EdgeColor','None')
    bar(1800+500*i,sqrt(mean((X(i,1:end-1)-Xff(i,:)).^2)),'BarWidth',100,'EdgeColor','None')
    grid on
end
title('Residuals')
axis([0 3000 -.02 .02])