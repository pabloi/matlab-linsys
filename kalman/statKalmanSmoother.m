function [Xs,Ps,Pt,Xf,Pf,rejSamples]=statKalmanSmoother(Y,A,C,Q,R,x0,P0,B,D,U,outRejFlag,constFun)
%Implements a Kalman smoother for a stationary system
%INPUT:
%Y: D1xN observed data
%U: D3xN input data
%A,C,Q,R,B,D: system parameters, B,D,U are optional (default=0)
%x0,P0: initial guess of state and covariance, optional
%outRejFlag: flag to indicate if outlier rejection should be performed
%constFun: function to enforce additional constraints on state estimates, see filterStationary_wConstraint()
%OUTPUT:
%Xs: D1xN, MLE estimate of state after smoothing
%Ps: D1xD1xN, covariance of state after smoothing
%Pt: D1xD1x(N-1) covariance of state transitions after smoothing
%Xf: D1xN, MLE estimate of state after FILTERING (i.e. forward pass only), see statKalmanFilter()
%Pf: D1xD1xN, covariance of state after FILTERING (i.e. forward pass only), see statKalmanFilter()
%See also:
% statKalmanFilter, filterStationary_wConstraint, trueEM

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
  if nargin<12 || isempty(constFun)
      constFunFlag=0;
  else
      constFunFlag=1;
  end

  %Size checks:
  %TODO

%Step 1: forward filter
if constFunFlag==0
    %[X,P,Xp,Pp,rejSamples]=filterStationary(Y,A,C,Q,R,x0,P0,B,D,U,outRejFlag);
    [Xf,Pf,Xp,Pp,rejSamples]=statKalmanFilter(Y,A,C,Q,R,x0,P0,B,D,U,outRejFlag);
else
    [Xf,Pf,Xp,Pp,rejSamples]=filterStationary_wConstraint(Y,A,C,Q,R,x0,P0,B,D,U,constFun);  
end

%Step 2: backward pass: (following the Rauch-Tung-Striebel implementation:
%https://en.wikipedia.org/wiki/Kalman_filter#Fixed-interval_smoothers)
Xs=Xf;
Ps=Pf;
prevXs=Xf(:,end);
prevPs=Pf(:,:,end);
%S=pinv(Q)*A;
D1=size(A,1);
Pt=nan(D1,D1,size(Y,2)-1); %Transition covariance matrix

for i=(size(Y,2)-1):-1:1
  
  %First, get estimates from forward pass:
  x=Xf(:,i); %Previous posterior estimate of covariance at this step
  PP=Pf(:,:,i); %Previous posterior estimate of covariance at this time step

  %Backward pass:
  AP=A*PP;
  APAQ=AP*A'+Q;
  %newK=lsqminnorm(APAQ,AP,1e-8)'; %Gain, eq to PP*A'/(A*PP*A'+Q)
  newK=AP'/APAQ; %Faster, although worse conditioned than line above
  newPs=PP + newK*(prevPs - APAQ)*newK';
  
  %Improved (smoothed) state estimate
  prevXs=x + newK*(prevXs-Xp(:,i+1)); 
  Xs(:,i)=prevXs;
  Pt(:,:,i)=prevPs'*newK';
  
  %Improved (smoothed) covariance estimate
  prevPs=newPs;
  Ps(:,:,i)=prevPs;
end

end
