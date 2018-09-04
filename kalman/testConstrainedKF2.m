%% Generate model: 3D movement of point-particle, cartesian coordinates
dT=.001; %time-interval for integration
A=[eye(3),dT*eye(3); zeros(3), eye(3)];
B=[dT^2/2 * eye(3); dT*eye(3)]; %input represents external forces
C=[eye(3) zeros(3)]; %We measure position only
D=[zeros(3)]; %No feed-through term
Q=1e-6*eye(6); %Force (impulse) measurement uncertainty
R=1e-4*eye(3); %Position measurement uncertainty

%% Constrain the particle to move in a circle: x^2+y^2=1, z=1
%Linearization: x'*x=1 -> 2*xo'*x =xo'*xo-1
%Additional: velocity in circle: v'*x=0
constrFun=@(x) deal([[x(1:2)'/sqrt(sum(x(1:2).^2)) zeros(1,4)];[0,0,1,0,0,0];[0,0,0,0,0,1]],[1;1;0]);

%% Simulate: (the constraining is done by other methods)
t=[0:7000-1]*dT;
U=zeros(3,7000);
Ua=[-cos(t);-sin(t);zeros(1,7000)]; %3 seconds of centripetal force, such that we get movement in a circle with v=1
x0=[1;0;1;0;1;0]; %Starts from (0,0,1), moving along x=-y direction
[Y,X]=fwdSim(Ua,A,B,C,D,x0,[],R);
%Notice that in this case the constraint is not enforced as such, but rather stems from the centripetal force input.
%This proves that constraining the Kalman filter may be a way to deal with unknown dynamics.
%In a sense, it is similar to Lagrangian mechanics: we forget about describing the reactive forces,
%and instead use the constraints implied by them to figure out the kinematics
norm(sum(X(1:2,:).^2,1)-1) %Verify state satisfies constraints
norm(X(3,:)-1) %Verify state satisfies constraints
%figure; hold on; plot3(Y(1,:),Y(2,:),Y(3,:),'.'); plot3(X(1,:),X(2,:),X(3,:),'LineWidth',2); axis equal %Plot

%% Kalman filter: (approximate params, unconstrained filter)
P0=eye(6); %uncertain data
outlierRejection=false;
[Xf,Pf,Xp,Pp,rejSamples]=statKalmanFilter(Y,A,C,Q,R,x0,P0,B,D,U,outlierRejection,[]);
[Xf,Pf,Xp,Pp,rejSamples]=statKalmanSmoother(Y,A,C,Q,R,x0,P0,B,D,U,outlierRejection,[]);

%% Constrained filter (true params)
[Xf2,Pf2,Xp,Pp,rejSamples]=statKalmanFilterConstrained(Y,A,C,Q,R,x0,P0,B,D,U,outlierRejection,constrFun);
[Xf2,Pf2,Xp,Pp,rejSamples]=statKalmanSmootherConstrained(Y,A,C,Q,R,x0,P0,B,D,U,outlierRejection,constrFun);
norm(sum(Xf2(1:2,:).^2)-1) %Verify state satisfies constraints

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
