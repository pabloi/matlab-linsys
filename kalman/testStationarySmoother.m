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
Xs=statKalmanSmoother(Y,A,C,Q,R,[],[],B,D,U,false); %Kalman smoother estimation of states, given the true parameters (this is the best possible estimation of states)
Xf=statKalmanFilter(Y,A,C,Q,R,[],[],B,D,U,false); 
%% Visualize results
figure
for i=1:2
    subplot(2,1,i)
    plot(Xs(i,:),'DisplayName','Smoothed')
    hold on
    plot(Xf(i,:),'DisplayName','Filtered')
    plot(X(i,:),'DisplayName','Actual')
    
    legend
    set(gca,'ColorOrderIndex',1)
    bar(1900,sqrt(mean((X(i,1:end-1)-Xs(i,:)).^2)),'BarWidth',100,'EdgeColor','None')
    bar(2000,sqrt(mean((X(i,1:end-1)-Xf(i,:)).^2)),'BarWidth',100,'EdgeColor','None')
end