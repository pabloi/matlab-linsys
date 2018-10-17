%% Example to test how EM behaves on a degenerate system with more states than output dimensions
%It works, but because of the degeneracy, it requires many initializations to converge to a good solution.
%The subspace-based initialization fails miserably at guessing a proper solution, as expected (it is based on outpud dimension)
clear all
%% Create model:
D1=2;
D2=2;
N=1000;
A=[.985,0;0,.995];
%B=3*randn(D1,1);
%B=B./sign(B); %Forcing all elements of B to be >0, WLOG
B=[.015; ,.005];
U=[zeros(300,1);ones(N,1);zeros(N/2,1)]'; %Step input and then removed
C=randn(D2,1)*ones(1,D2);
C=C+.01*randn(size(C)); %Almost colinear C
%C=randn(D2,D1);
D=randn(D2,1);
Q=eye(D1)*.0001;
R=eye(D2)*.01;
%% Simulate
NN=size(U,2);
x0=zeros(D1,1);
[Y,X]=fwdSim(U,A,B,C,D,x0,Q,R);
[Xs,Ps]=statKalmanSmoother(Y,A,C,Q,R,[],[],B,D,U,false); %Kalman smoother estimation of states, given the true parameters (this is the best possible estimation of states)
logL=dataLogLikelihood(Y,U,A,B,C,D,Q,R,Xs(:,1),Ps(:,:,1))
[A,B,C,Xs,~,Q,Ps] = canonize(A,B,C,Xs,Q,Ps);
logL=dataLogLikelihood(Y,U,A,B,C,D,Q,R,Xs(:,1),Ps(:,:,1))

%Save model:
J=A;
model{1}=autodeal(J,B,C,D,X,Q,R,logL);
model{1}.name='true';

%% Identify with original scale:
opts.fastFlag=true;
opts.convergenceTol=1e-6;
opts.Niter=1000;
tic
[A,B,C,D,Q,R,X,P,logL]=randomStartEM(Y,U,D1,5,opts);
[J,B,C,X,~,Q,P] = canonize(A,B,C,X,Q,P);
model{2}=autodeal(J,B,C,D,X,Q,R,logL);
model{2}.name='EM (fast, original scale)';
toc
%% Compare:
vizModels(model)
vizDataFit(model,Y,U)
