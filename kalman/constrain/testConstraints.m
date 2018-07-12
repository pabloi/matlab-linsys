%% Generate data
N=1e3;
X=nan(2,N);
Y=nan(2,N);
X(:,1)=[1;0];
Y(:,1)=[1;0];
alpha=.1;
A=[cos(alpha) sin(alpha); -sin(alpha) cos(alpha)];
R=.1*eye(2);
r=sqrtm(R);
C=eye(2);
for i=2:N
  X(:,i)=A*X(:,i-1); %Real Q is 0!
  Y(:,i)=C*X(:,i)+ r*randn(2,1);
end

%% Add S&P noise
idx=randi(N,20);
Y(:,idx)=5;

%% Run filter w/o constraint
Q=.01*eye(2);
q=sqrtm(Q);
outRej=0; %No outlier rejection
x0=[1;0];
P0=1e3*ones(2);
B=[0;0];
D=[0;0];
U=zeros(1,N);
[Xs,Ps,Xp,Pp,rejSamples]=filterStationary(Y,A,C,Q,R,x0,P0,B,D,U,outRej);

%% Filter outliers
outRej=1;
[Xs3,Ps3,Xp3,Pp3,rejSamples]=filterStationary(Y,A,C,Q,R,x0,P0,B,D,U,outRej);

%% Smooth w/outlier rejection
[Xs4,Ps4,Xa,Pa,Xp,Pp,rejSamples]=smoothStationary(Y,A,C,Q,R,x0,P0,B,D,U,outRej);
%% Run filter adding constraint
constFun=@(x) circleConstraint(x);
[Xs2,Ps2,Xp2,Pp2,rejSamples]=filterStationary_wConstraint(Y,A,C,Q,R,x0,P0,B,D,U,constFun);

%% Smooth w/outlier rejection & constraint
[Xs5,Ps5,Xa5,Pa5,Xp5,Pp5,rejSamples]=smoothStationary(Y,A,C,Q,R,x0,P0,B,D,U,outRej,constFun);

%% Compare & contrast
figure; plot(Y(1,:),Y(2,:)); hold on; plot(Xs(1,:),Xs(2,:)); plot(Xs2(1,:),Xs2(2,:)); plot(Xs3(1,:),Xs3(2,:));plot(Xs4(1,:),Xs4(2,:)); plot(Xs5(1,:),Xs5(2,:))
figure; hold on; histogram(Xs(1,:)-X(1,:)); histogram(Xs2(1,:)-X(1,:)); histogram(Xs3(1,:)-X(1,:)); histogram(Xs4(1,:)-X(1,:));histogram(Xs5(1,:)-X(1,:));