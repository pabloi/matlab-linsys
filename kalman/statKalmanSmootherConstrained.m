function [Xs,Ps,Pt,Xf,Pf,Xp,Pp,rejSamples]=statKalmanSmootherConstrained(Y,A,C,Q,R,x0,P0,B,D,U,outRejFlag,constrFun)
%Implements a Kalman smoother for a stationary system
%INPUT:
%Y: D1xN observed data
%U: D3xN input data
%A,C,Q,R,B,D: system parameters, B,D,U are optional (default=0)
%x0,P0: initial guess of state and covariance, optional
%outRejFlag: flag to indicate if outlier rejection should be performed
%constrFun: additional constraint function to be enforced for states
%OUTPUT:
%Xs: D1xN, MLE estimate of state after smoothing
%Ps: D1xD1xN, covariance of state after smoothing
%Pt: D1xD1x(N-1) covariance of state transitions after smoothing
%Xf: D1xN, MLE estimate of state after FILTERING (i.e. forward pass only), see statKalmanFilter()
%Pf: D1xD1xN, covariance of state after FILTERING (i.e. forward pass only), see statKalmanFilter()
%See also:
% statKalmanFilterConstrained, statKalmanSmoother

[D2,N]=size(Y);
%Init missing params:
if nargin<6 || isempty(x0)
x0=zeros(size(A,1),1); %Column vector
end
if nargin<7 || isempty(P0)
P0=1e8 * eye(size(A));
end
if nargin<8 || isempty(B)
B=0;
end
if nargin<9 || isempty(D)
D=0;
end
if nargin<10 || isempty(U)
 U=zeros(1,size(Y,2));
end
if nargin<11 || isempty(outRejFlag)
  outRejFlag=0;
end
%Size checks:
%TODO

%Step 1: forward filter
[Xf,Pf,Xp,Pp,rejSamples]=statKalmanFilterConstrained(Y,A,C,Q,R,x0,P0,B,D,U,outRejFlag,constrFun);

%Step 2: backward pass: (following the Rauch-Tung-Striebel implementation:
%https://en.wikipedia.org/wiki/Kalman_filter#Fixed-interval_smoothers)
Xs=Xf;
Ps=Pf;
prevXs=Xf(:,end);
prevPs=Pf(:,:,end);
D1=size(A,1);
Pt=nan(D1,D1,size(Y,2)-1); %Transition covariance matrix

%Do true smoothing for first M samples:
for i=(N-1):-1:1
  %First, get estimates from forward pass:
  xf=Xf(:,i); %Previous posterior estimate of covariance at this step
  pf=Pf(:,:,i); %Previous posterior estimate of covariance at this time step
  xp=Xp(:,i+1); %Prediction of next step based on post estimate of this step
  pp=Pp(:,:,i+1); %Covariance of next step based on post estimate of this step

  %Backward pass:
  %First, compute gain:
  AP=A*pf;
  %pp=AP*A'+Q; %Could compute pp instead of accessing it, unclear which is faster
  newK=AP'/pp; %Faster, although worse conditioned than: newK=lsqminnorm(pp,AP,1e-8)'

  %Improved (smoothed) state estimate
  newPt=prevPs*newK'; %This should be such that A*newPt' is hermitian
  %newPs=pf+newK*(newPt-AP); %=newK*prevPs'*newK' + pf -pf'*(A'/pp)*A*pf =  newK*prevPs'*newK' + pf -((pf'*A')/(A*pf*A'+Q))*A*pf = newK*prevPs'*newK' + inv(inv(pf) +A'*inv(Q)*A) =
  sPs=cholcov(prevPs); %Cholcov permits semi-definite matrices, but is 10x slower than chol()
  Kps=newK*sPs';
  sPr=cholcov(newK*AP); %newK*AP= ((pf'*A')/(A*pf*A'+Q))*A*pf
  newPs=Kps*Kps' + pf -sPr'*sPr;  %Would it be more precise/efficient to compute the sum of the last two terms as inv(inv(pf) +A'*inv(Q)*A) ?

  %xp=A*xf+B*U(:,i); %Could compute instead of acccesing it, unclear which is faster
  prevXs=xf + newK*(prevXs-xp);

  Xs(:,i)=prevXs;
  Pt(:,:,i)=newPt;

  %Improved (smoothed) covariance estimate
  prevPs=newPs;
  Ps(:,:,i)=prevPs;
end

end
