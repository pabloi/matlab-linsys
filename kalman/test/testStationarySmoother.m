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
addpath(genpath('../aux/'))
NN=size(U,2);
x0=zeros(D1,1);
[Y,X]=fwdSim(U,A,B,C,D,x0,Q,R);
[Y1,X1]=fwdSim(U,A,B,C,D,x0,[],[]); %Noiseless simulation, for comparison

%% Do kalman smoothing with true params
tic
fastFlag=[];
fastFlag=0;
[Xs,Ps,Pt,Xf,Pf,Xp,Pp]=statKalmanSmoother(Y,A,C,Q,R,[],[],B,D,U,false,fastFlag); %Kalman smoother estimation of states, given the true parameters (this is the best possible estimation of states)
tf=toc;
%% Use Cheng & Sabes code:
addpath(genpath('../ext/lds-1.0/'))
LDS.A=A;
LDS.B=B;
LDS.C=C;
LDS.D=D;
LDS.Q=Q;
LDS.R=R;
LDS.x0=zeros(D1,1);
LDS.V0=1e8 * eye(size(A)); %Same as my smoother uses 
tic
[Lik,Xcs,Pcs,Ptcs,Scs] = SmoothLDS(LDS,Y,U,U); 
tc=toc;
%% Visualize results
figure
for i=1:2
    subplot(3,1,i)
    plot(Xs(i,:),'DisplayName','Smoothed')
    hold on
    plot(Xf(i,:),'DisplayName','Filtered')
    plot(Xcs(i,:),'DisplayName','CS2006')
    plot(X(i,:),'DisplayName','Actual')
    
    legend
    if i==1
        title(['This runtime= ' num2str(tf) ', C&S2006 runtime= ' num2str(tc)]);
    end
end
subplot(3,1,3)
for i=1:2
     hold on
     set(gca,'ColorOrderIndex',1)
    plot(Xs(i,:)-X(i,1:end-1),'DisplayName','Smoothed')
    plot(Xf(i,:)-X(i,1:end-1),'DisplayName','Filtered')
    plot(Xcs(i,:)-X(i,1:end-1),'DisplayName','CS2006')
        set(gca,'ColorOrderIndex',1)
        aux=sqrt(mean((X(i,1:end-1)-Xs(i,:)).^2));
    b1=bar(1900+400*i,aux,'BarWidth',100,'EdgeColor','None');
    text(1850+400*i,1.2*aux,num2str(aux,3),'Color',b1.FaceColor)
    aux=sqrt(mean((X(i,1:end-1)-Xf(i,:)).^2));
    b1=bar(2000+400*i,aux,'BarWidth',100,'EdgeColor','None');
    text(1950+400*i,1.4*aux,num2str(aux,3),'Color',b1.FaceColor)
    aux=sqrt(mean((X(i,1:end-1)-Xcs(i,:)).^2));
    b1=bar(2100+400*i,aux,'BarWidth',100,'EdgeColor','None');
    text(2050+400*i,1.2*aux,num2str(aux,3),'Color',b1.FaceColor)
    grid on
end
title('Residuals')
axis([0 3000 -.02 .02])