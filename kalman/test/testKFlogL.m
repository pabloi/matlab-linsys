%test kalmanFilter logL
if isOctave
pkg load statistics %Needed in octave to use nanmean()
end
[folder,fname,ext]=fileparts(mfilename('fullpath'));
addpath([folder '/../../misc/'])
%% Create model:
D1=2;
D2=100;
N=1000;
A=[.95,0;0,.99];
%B=B./sign(B); %Forcing all elements of B to be >0, WLOG
B=(eye(size(A))-A)*ones(size(A,1),1); %WLOG, arbitrary scaling
U=[zeros(300,1);ones(N,1);zeros(N/2,1)]'; %Step input and then removed
C=randn(D2,D1);
D=randn(D2,1);
Q=eye(D1)*.0005;
R=eye(D2)*.01;

%% Simulate
addpath(genpath('../sim/'))
NN=size(U,2);
x0=zeros(D1,1);
[Y,X]=fwdSim(U,A,B,C,D,x0,Q,R);
[Y1,X1]=fwdSim(U,A,B,C,D,x0,[],[]); %Noiseless simulation, for comparison

%% Do kalman smoothing with true params
tic
opts.noReduceFlag=false;
[Xf,Pf,Xp,Pp,rejectedSamples,logL]=statKalmanFilter(Y,A,C,Q,R,[],[],B,D,U,opts); %Kalman smoother estimation of states, given the true parameters (this is the best possible estimation of states)
tc=toc;
disp(['KF logL, no reduced model: ' num2str(logL) ', runtime: ' num2str(tc) 's.'])
tic
opts.noReduceFlag=true;
[Xf1,Pf1,Xp1,Pp1,rejectedSamples,logL1]=statKalmanFilter(Y,A,C,Q,R,[],[],B,D,U,opts); %Kalman smoother estimation of states, given the true parameters (this is the best possible estimation of states)
tc=toc;
disp(['KF logL, reduced model: ' num2str(logL1) ', runtime: ' num2str(tc) 's.'])
disp(['KF logL difference: ' num2str(logL1-logL)])

tic
logLperSamplePerDim=dataLogLikelihood(Y,U,A,B,C,D,Q,R,Xp,Pp,'exact');
tc=toc;
disp(['logL exact, legacy computation, non-reduced filtered data: ' num2str(logLperSamplePerDim) ', runtime: ' num2str(tc) 's.'])
tic
logLperSamplePerDim=dataLogLikelihood(Y,U,A,B,C,D,Q,R,Xp1,Pp1,'exact');
tc=toc;
disp(['logL exact, legacy computation, reduced filtered data: ' num2str(logLperSamplePerDim) ', runtime: ' num2str(tc) 's.'])
tic
logLperSamplePerDim=dataLogLikelihood(Y,U,A,B,C,D,Q,R,Xp1,Pp1,'approx');
tc=toc;
disp(['logL approx, legacy computation, reduced filtered data: ' num2str(logLperSamplePerDim) ', runtime: ' num2str(tc) 's.'])
