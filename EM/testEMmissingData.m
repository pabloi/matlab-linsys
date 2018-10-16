ny=100;
A=[.95,0;0,.99];
B=[.05;.01];
C=randn(ny,2);
D=randn(ny,1);
x0=zeros(1,2);
Q=1e-3*eye(2);
R=1e-3*eye(ny);
U=[zeros(1,300) ones(1,1000) zeros(1,500)];
[Y,~]=fwdSim(U,A,B,C,D,x0,[],R);
[X,P,Pt,Xf,Pf,Xp,Pp,rejSamples]=statKalmanSmoother(Y,A,C,Q,R,[],[],B,D,U);

%% Estimate params from full data:
opts.fastFlag=true;
opts.Niter=1000;
%[A1,B1,C1,D1,Q1,R1,x01,P01]=estimateParams(Y,U,X,P,Pt,opts);
[A1,B1,C1,D1,Q1,R1,X1,P1,bestLogL1]=EM(Y,U,2,opts,[]);

%% Estimate params from half data:
Ymissing=Y;
Ymissing(:,2:2:end)=NaN;
%[A2,B2,C2,D2,Q2,R2,x02,P02]=estimateParams(Ymissing,U,X,P,Pt,processEMopts([]));
[A2,B2,C2,D2,Q2,R2,X2,P2,bestLogL2]=EM(Ymissing,U,2,opts);

%% Estimate params from other half data:
Ymissing=Y;
Ymissing(:,1:2:end)=NaN;
%[A3,B3,C3,D3,Q3,R3,x03,P03]=estimateParams(Ymissing,U,X,P,Pt,processEMopts([]));
[A3,B3,C3,D3,Q3,R3,X3,P3,bestLogL3]=EM(Ymissing,U,2,opts);
