%%
A=[[.5,0,0]; [0,.7,0];[ 0 ,0, .99]];
B=[1;1;1]; %Both states asymptote at 1
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
[Y,X]=fwdSim(U,A,B,C,D,x0,Q,R);
Xn=X+.3*randn(size(X));
%%
N=16;
v1=nan(numel(x0),N/2);
v2=nan(numel(x0),N/2);
v3=nan(numel(x0),N/2);
for d=1:2:N %Can only estimate for odd
  A1=estimateTransitionMatrix(X(:,1:end-1)-(X(:,1:end-1)/U)*U,d);
  %A1=estimateTransitionMatrix(X,d);
  v1(:,(d+1)/2)=sort(eig(A1));
  An=estimateTransitionMatrix(Xn(:,1:end-1)-(Xn(:,1:end-1)/U)*U,d);
  v2(:,(d+1)/2)=sort(eig(An));
  %An2=estimateTransitionMatrix(Xn(:,1:end-1)-(Xn(:,1:end-1)/U)*U,[1 zeros(1,d-1)]);
  %An2=(Xn(:,2:end)-(Xn(:,2:end)/U)*U)/Xn(:,1:end-1);
  %v3(:,(d+1)/2)=sort(eig(An2));
end

%%
figure
for i=1:numel(x0)
  subplot(numel(x0),1,i)
  plot(abs(v1(i,:)))
  hold on
  plot(abs(v2(i,:)))
  plot(abs(v3(i,:)))
  %set(gca,'YScale','log','YLim',[.8 1.01])
  grid on
end
