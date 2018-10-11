%%
A=[[.5,0,0]; [0,.7,0];[ 0 ,0, .9]];
B=[0;0;0]; %Both states asymptote at 1
B=zeros(3,1);
Ny=3;
C=randn(Ny,3);
D=randn(Ny,1);
Q=.01*eye(3);
R=.1*eye(Ny);
x0=[1;1;1];
U=[zeros(1,500), ones(1,1000)];
d=2;
%%
Niter=1e2;
N=20;
s=.3;
v1=nan(numel(x0),N/2,Niter);
v2=nan(numel(x0),N/2,Niter);
v3=nan(numel(x0),N/2,Niter);
[Y,X]=fwdSim(U,A,B,C,D,x0,Q,R);
for k=1:Niter
  Xn=X+s*randn(size(X));
  for d=1:2:N %Can only estimate for odd
    %A1=estimateTransitionMatrix(X(:,1:end-1)-(X(:,1:end-1)/U)*U,d);
    A1=estimateTransitionMatrix(X,d);
    v1(:,(d+1)/2,k)=sort(eig(A1));
    %An=estimateTransitionMatrix(Xn(:,1:end-1)-(Xn(:,1:end-1)/U)*U,d);
    An=estimateTransitionMatrix(Xn(:,1:end-1),d);
    v2(:,(d+1)/2,k)=sort(eig(An));
    An2=real(estimateTransitionMatrix(Xn(:,1:end-1),[1 zeros(1,d-1)]));
    %An2=(Xn(:,2:end)-(Xn(:,2:end)/U)*U)/Xn(:,1:end-1);
    v3(:,(d+1)/2,k)=sort(eig(An2));
  end
end

%%
figure
for d=1:2:N %Can only estimate for odd
  An2_theoretical=A*matrixPolyRoots(eye(size(A))-size(X,2)*s^2*eye(size(A))/(X*X'+size(X,2)*s^2*eye(size(A))),[1 zeros(1,d-1)]);
  v3_theo(:,(d+1)/2)=sort(eig(An2_theoretical));
end
for i=1:numel(x0)
  subplot(numel(x0),1,i)
  plot(1:2:N,abs(mean(v1(i,:,:),3)),'DisplayName','Noiseless estimate from A+A^2+...+A^d')
  hold on
  plot(1:2:N,abs(mean(v2(i,:,:),3)),'DisplayName','Noisy estimate from A+A^2+...+A^d')
  plot(1:2:N,abs(mean(v3(i,:,:),3)),'DisplayName','Noisy estimate from A^d')
  plot(1:2:N,abs(v3_theo(i,:)),'DisplayName','Theoretical expected estimate from A^d')
  %set(gca,'YScale','log','YLim',[.8 1.01])
  grid on
  xlabel('d (estimate order)')
  ylabel(['|E(\lambda_' num2str(i) ')|'])
  if i==numel(x0)
    legend('Location','SouthEast')
  end
end
