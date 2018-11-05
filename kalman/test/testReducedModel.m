%test kalmanFilter logL
if isOctave
pkg load statistics %Needed in octave to use nanmean()
end
[folder,fname,ext]=fileparts(mfilename('fullpath'));
addpath([folder '/../../misc/'])
%% Create model:
D1=2;
D2=200;
N=1000;
A=[.95,0;0,.99];
%B=B./sign(B); %Forcing all elements of B to be >0, WLOG
B=(eye(size(A))-A)*ones(size(A,1),1); %WLOG, arbitrary scaling
U=[zeros(300,1);ones(N,1);zeros(N/2,1)]'; %Step input and then removed
C=randn(D2,D1);
D=zeros(D2,1);
Q=.1*randn(D1);
Q=Q'*Q;
R=eye(D2)*.01;

%% Simulate
addpath(genpath('../sim/'))
NN=size(U,2);
x0=zeros(D1,1);
[Y,X]=fwdSim(U,A,B,C,D,x0,Q,R);
[Y1,X1]=fwdSim(U,A,B,C,D,x0,[],[]); %Noiseless simulation, for comparison

%% Full model logL:
samp=1;
[l0,z0]=logLnormal(Y,R+C*Q*C');

%%Reduced model logL:
[Cnew,Rnew,Ynew,cR,logLmargin,logLDetMargin,dimMargin,z2Margin]=reduceModel(C,R,Y);
[l,z]=logLnormal(Ynew,Rnew+Cnew*Q*Cnew');
correctedL=(l+logLmargin);
correctedZ=z+z2Margin;

meanAbsErrorZ=mean(abs(z0-correctedZ));
meanAbsErrorL=mean(abs(l0-correctedL));

disp(['Mean-abs error logL reduced (corrected) vs. non-reduced: ' num2str(meanAbsErrorL)])
disp(['Mean-abs error z^2 score reduced (corrected) vs. non-reduced: ' num2str(meanAbsErrorZ)])
