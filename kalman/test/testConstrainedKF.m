%% Generate model: 3D movement of point-particle, cartesian coordinates
dT=.001; %time-interval for integration
A=[eye(3),dT*eye(3); zeros(3), eye(3)];
B=[dT^2/2 * eye(3); dT*eye(3)]; %input represents external forces
C=[eye(3) zeros(3)]; %We measure position only
D=[zeros(3)]; %No feed-through term
Q=1e-6*eye(6); %Force (impulse) measurement uncertainty
R=1e-6*eye(3); %Position measurement uncertainty

%% Add constraints (if any(b~0) this requires augmenting the state vector to simulate):
H=[ones(1,3) zeros(1,3);zeros(1,3) ones(1,3)]; %Constraint to impose, H*x=b;
b=[1;0]; %Constraining to plane x+y+z=1, and vx+vy+vz=0, so particle never escapes plane
pH=pinv(H); %Requirement is that H*pH = eye()
G=eye(size(A))- pH*H;
Aa=[G*A pH*b;zeros(1,size(A,2)) 1]; %Augmented A
Ba=[G*B; zeros(1,size(B,2))]; %Augmented B
Qb=G*Q*G';%+1e-15*eye(size(Q)); %Modified Q
Qa=[Qb zeros(size(Qb,1),1);zeros(1,size(Qb,1)) 0]; %Augmented Q
Ca=[C zeros(size(C,1),1)];

%% Simulate:
U=zeros(3,100); %.1 second of null force
x0=[0;0;1;-1;1;0]; %Starts from (0,0,1), moving along x=-y direction
[Y,X]=fwdSim(U,Aa,Ba,Ca,D,[x0;1],Qa,R);
%In this very particular case (because U=0, initial conditions satisfy constraints,
%and evolution of system also satisfies constraints) this simulationcan be done
%without augmenting the states, and solely constraining Q (i.e. using A,B,C and Qb)
norm(H*X(1:end-1,:)-b) %Verify state satisfies constraints

%% Kalman filter: (approximate params, unconstrained filter)
P0=eye(6); %uncertain data
outlierRejection=false;
[Xf,Pf,Xp,Pp,rejSamples]=statKalmanFilter(Y,A,C,Q,R,x0,P0,B,D,U,outlierRejection,[]);
[Xf,Pf,Xp,Pp,rejSamples]=statKalmanSmoother(Y,A,C,Q,R,x0,P0,B,D,U,outlierRejection,[]);

%% Constrained filter (true params)
constrFun=@(x) deal(H,b);
[Xf2,Pf2,Xp,Pp,rejSamples]=statKalmanFilterConstrained(Y,A,C,Q,R,x0,P0,B,D,U,outlierRejection,constrFun);
[Xf2,Pf2,Xp,Pp,rejSamples]=statKalmanSmootherConstrained(Y,A,C,Q,R,x0,P0,B,D,U,outlierRejection,constrFun);
norm(H*Xf2-b) %Verify filtered state satisfies constraints

%% Plot
figure
subplot(2,2,1)
hold on
plot3(Y(1,:),Y(2,:),Y(3,:),'k.')
plot3(X(1,:),X(2,:),X(3,:),'k')
plot3(Xf(1,:),Xf(2,:),Xf(3,:),'r')
plot3(Xf2(1,:),Xf2(2,:),Xf2(3,:),'g')
h1=quiver3(X(1,:),X(2,:),X(3,:),X(4,:),X(5,:),X(6,:),0,'b') ;%True data
h2=quiver3(Xf(1,:),Xf(2,:),Xf(3,:),Xf(4,:),Xf(5,:),Xf(6,:),0,'r') ;%Estimated data
axis equal

scale=.002;
hU1 = get(h1,'UData');
hV1 = get(h1,'VData');
hW1 = get(h1,'WData');
set(h1,'UData',scale*hU1,'VData',scale*hV1,'WData',scale*hW1)
hU2 = get(h2,'UData');
hV2 = get(h2,'VData');
hW2 = get(h2,'WData');
set(h2,'UData',scale*hU2,'VData',scale*hV2,'WData',scale*hW2)
view(2)
legend('Measured data','True states','Standard Kalman filter','Constrained Kalman filter')
for i=1:3
  subplot(2,2,i+1)
  idx1=mod(i-1,3)+1;
  idx2=mod(i,3)+1;
  hold on
  plot(Y(idx1,:),Y(idx2,:),'k.')
  plot(X(idx1,:),X(idx2,:),'k')
  plot(Xf(idx1,:),Xf(idx2,:),'r')
  plot(Xf2(idx1,:),Xf2(idx2,:),'g')
  h1=quiver(X(idx1,:),X(idx2,:),X(idx1+3,:),X(idx2+3,:),0,'b') ;%True data
  h2=quiver(Xf(idx1,:),Xf(idx2,:),Xf(idx1+3,:),Xf(idx2+3,:),0,'r') ;%Estimated data
  hU1 = get(h1,'UData');
  hV1 = get(h1,'VData');
  set(h1,'UData',scale*hU1,'VData',scale*hV1)
  hU2 = get(h2,'UData');
  hV2 = get(h2,'VData');
  set(h2,'UData',scale*hU2,'VData',scale*hV2)
  xlabel(['x_' num2str(idx1)])
  ylabel(['x_' num2str(idx2)])
end
