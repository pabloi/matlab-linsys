%% Testing EM identification and model selection criteria when true model is non-lin
clear all
addpath(genpath('../../'))
%% Simulate model 1: single-state tracker

N=1800;
U=[zeros(300,1);ones(900,1);zeros(600,1)]'; %Step input and then removed
M=100;
D1=2;
C=randn(M,D1);  %Making the output multi-dimensional helps in convergence, because a single sample is sufficient to get a non-zero estimate in all dimensions of covariance. 
%Otherwise, the combination of this and infinite initial uncertainty forces the system to deal with ill-conditioned situations for some steps.
D=randn(M,1);
x=nan(D1,N);
y=nan(size(C,1),N);
x(:,1)=0;
y(:,1)=0;
for i=2:N
    e=sum(x(:,i-1))-U(i-1); %Error is the difference between past state and past input. 
    e2=sign(e)*e.^2; %This will allow the state to track the error with quadratic corrections 
    x(1,i)=.999*x(1,i-1)-.01*e2+.005*randn;
    x(2,i)=.9*x(2,i-1)-.1*e2+.005*randn;
    y(:,i)=C*x(:,i)+D*U(i)+.3*randn(M,1); 
end

figure; plot(x'); hold on; plot(U); plot(sum(x)-U)
%figure; plot(y')
%% Identify:
ds=dset(U,y);
opts.fastFlag=true;
opts.Nreps=5;
opts.Niter=1e3;
opts.includeOutputIdx=[];
opts.stableA=true;
mdl=linsys.id(ds,0:4,opts); %Identifying with model orders from 0 to 3
save EMnonlin1.mat ds mdl
mdl=cellfun(@(x) x.canonize,mdl,'UniformOutput',false);
ds.vizFit(mdl)
fittedLinsys.compare(mdl)
%% Simulate model 2: generalized linear model with quadratic obs

%Do fwdSim:
D1=2; %2 states
D2=100; %100 outputs
A=[.98,0; 0,.998]; %50,500 time constants
%B=B./sign(B); %Forcing all elements of B to be >0, WLOG
B=(eye(size(A))-A)*ones(size(A,1),1); %WLOG, arbitrary scaling
U=[zeros(300,1);ones(900,1);zeros(600,1)]'; %Step input and then removed
C=randn(D2,D1);
D=randn(D2,1);
Q=eye(D1)*.0003;
R=eye(D2)*.1;
NN=size(U,2);
x0=zeros(D1,1);
[Y,X]=fwdSim(U,A,B,C,D,x0,Q,R);
truMdl=linsys(A,C,R,B,D,Q);

%Non-linear out (preserving sign):
Y=sign(Y).*Y.^2;
figure; plot(X')
figure; plot(Y')

%% Identify:
ds2=dset(U,Y);
opts.fastFlag=true;
opts.Nreps=5;
opts.Niter=1e3;
opts.includeOutputIdx=[];
opts.stableA=true;
mdl2=linsys.id(ds2,0:4,opts); %Identifying with model orders from 0 to 3
save EMnonlin2.mat ds2 mdl2
mdl2=cellfun(@(x) x.canonize,mdl2,'UniformOutput',false);
ds2.vizFit(mdl2(2:end))
fittedLinsys.compare(mdl2(2:end))
figure; plot(mdl2{3}.Ksmooth(ds2).state'); hold on; plot(X')
linsys.vizMany([mdl2;  {truMdl}])