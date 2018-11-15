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
U=[zeros(300,1);ones(N,1);zeros(N/2,1)]'; %Step input and then removed
C=randn(D2,D1);
D=randn(D2,1);
Q=eye(D1)*.0005;
R=eye(D2)*.01;

%% Simulate
NN=size(U,2);
x0=zeros(D1,1);
[Y,X]=fwdSim(U,A,B,C,D,x0,Q,R);
[Y1,X1]=fwdSim(U,A,B,C,D,x0,[],[]); %Noiseless simulation, for comparison

%% Do kalman smoothing with true params
tic
opts.fastFlag=1; %fast, self-select samples
[Xsf,Psf,Ptf,Xff,Pff,Xpf,Ppf]=statKalmanSmoother(Y,A,C,Q,R,[],[],B,D,U,opts); %Kalman smoother estimation of states, given the true parameters (this is the best possible estimation of states)
tf=toc;
tic
opts.fastFlag=0; %No fast
[Xs,Ps,Pt,Xf,Pf,Xp,Pp]=statKalmanSmoother(Y,A,C,Q,R,[],[],B,D,U,opts); %Kalman smoother estimation of states, given the true parameters (this is the best possible estimation of states)
ts=toc;
tic
opts.fastFlag=30; %Forcing fast samples
[Xsff,Ps,Pt,Xfff,Pf,Xp,Pp]=statKalmanSmoother(Y,A,C,Q,R,[],[],B,D,U,opts); %Kalman smoother estimation of states, given the true parameters (this is the best possible estimation of states)
tff=toc;

%% Visualize results
figure
for i=1:2
    subplot(3,1,i)
    plot(Xs(i,:),'DisplayName','Smoothed')
    hold on
    plot(Xsf(i,:),'DisplayName','Fast Smooth')
    plot(Xsff(i,:),'DisplayName','Forced Fast Smooth')
    plot(X(i,:),'DisplayName','Actual')
    
    legend
    if i==1
        title(['Fast runtime= ' num2str(tf) ', Regular runtime= ' num2str(ts) ', Forced fast runtime= ' num2str(tff)]);
    end

end
subplot(3,1,3)
for i=1:2
     hold on
     set(gca,'ColorOrderIndex',1)
    plot(Xs(i,:)-X(i,1:end-1),'DisplayName','SlowSmoothed')
    plot(Xsf(i,:)-X(i,1:end-1),'DisplayName','FastSmooth')
    plot(Xsff(i,:)-X(i,1:end-1),'DisplayName','FastForced')
        set(gca,'ColorOrderIndex',1)
    %bar(1500+500*i,sqrt(mean((X(i,1:end-1)-Xs(i,:)).^2)),'BarWidth',100,'EdgeColor','None')
    aux=sqrt(mean((X(i,1:end-1)-Xs(i,:)).^2));
    b1=bar(1500+500*i,aux,'BarWidth',100,'EdgeColor','None');
    text(1450+500*i,1.2*aux,num2str(aux,3),'Color',b1.FaceColor)
    aux=sqrt(mean((X(i,1:end-1)-Xsf(i,:)).^2));
    b1=bar(1600+500*i,aux,'BarWidth',100,'EdgeColor','None');
    text(1550+500*i,1.2*aux,num2str(aux,3),'Color',b1.FaceColor)
    aux=sqrt(mean((X(i,1:end-1)-Xsff(i,:)).^2));
    b1=bar(1700+500*i,aux,'BarWidth',100,'EdgeColor','None');
    text(1650+500*i,1.2*aux,num2str(aux,3),'Color',b1.FaceColor)

    grid on
end
title('Residuals')
axis([0 3000 -.02 .02])