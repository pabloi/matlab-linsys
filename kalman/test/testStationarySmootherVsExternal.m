%% Create model:
D1=2;
D2=100;%100; 
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
U=[zeros(300,1);ones(N,1);zeros(N/2,1)]'; %Step input and then removed
C=randn(D2,D1);
D=randn(D2,1);
Q=eye(D1)*1e-1;
R=1*eye(D2); %CS2006 performance degrades (larger state estimation errors) for very small R

%% Simulate
NN=size(U,2);
x0=zeros(D1,1);
[Y,X]=fwdSim(U,A,B,C,D,x0,Q,R);
[Y1,X1]=fwdSim(U,A,B,C,D,x0,[],[]); %Noiseless simulation, for comparison

%% Do kalman smoothing with true params
tic
fastFlag=[];
opts.fastFlag=0;
[Xs,Ps,Pt,Xf,Pf,Xp,Pp]=statKalmanSmoother(Y,A,C,Q,R,[],[],B,D,U,opts); %Kalman smoother estimation of states, given the true parameters (this is the best possible estimation of states)
tf=toc;
%% Use Info smoother:
tic;
[Xf,Pf]=statInfoSmoother(Y,A,C,Q,R,[],[],B,D,U,opts); 
ts=toc;
%% Use Cheng & Sabes code:
[folder]=fileparts(mfilename('fullpath'));
addpath(genpath([folder '/../../../ext/lds-1.0/']))
LDS.A=A;
LDS.B=B;
LDS.C=C;
LDS.D=D;
LDS.Q=Q;
LDS.R=R;
LDS.x0=zeros(D1,1);
LDS.V0=1e8 * eye(size(A)); %My filter uses Inf as initial uncertainty, but CS2006 does not support it, or anything too large
warning('off')
tic
[Lik,Xcs,Pcs,Ptcs,Scs] = SmoothLDS(LDS,Y,U,U); %Mex version
tc=toc;
%tic
%[Lik,Xcs,Pcs,Ptcs,Scs] = SmoothLDS_(LDS,Y,U,U); %Matlab version
%tc=toc
%% Visualize results
figure
for i=1:2
    subplot(3,1,i)
    plot(Xs(i,:),'DisplayName','Smoothed','LineWidth',2)
    hold on
    plot(Xf(i,:),'DisplayName','InfoSmoothed','LineWidth',2)
    plot(Xcs(i,:),'DisplayName','CS2006','LineWidth',2)
    plot(X(i,:),'DisplayName','Actual','LineWidth',2)

    legend
    if i==1
        title(['This runtime= ' num2str(tf) ', info runtime= ' num2str(ts) ', C&S2006 runtime= ' num2str(tc)]);
    end
end
subplot(3,1,3)
for i=1:2
     hold on
     set(gca,'ColorOrderIndex',1)
    plot(Xs(i,:)-X(i,1:end-1),'DisplayName','Smoothed')
    plot(Xf(i,:)-X(i,1:end-1),'DisplayName','InfoSmoothed')
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
axis tight
