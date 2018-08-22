function [Xs,Ps,Pt,Xf,Pf,Xp,Pp,rejSamples]=statKalmanSmootherFast(Y,A,C,Q,R,x0,P0,B,D,U,outRejFlag,constFun)
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

%Size checks:
%TODO
  
%Step 1: forward filter
if nargin<12 || isempty(constFun)
  %[X,P,Xp,Pp,rejSamples]=filterStationary(Y,A,C,Q,R,x0,P0,B,D,U,outRejFlag);
  %[Xf,Pf,Xp,Pp,rejSamples]=statKalmanFilter(Y,A,C,Q,R,x0,P0,B,D,U,outRejFlag);
  [Xf,Pf,Xp,Pp,rejSamples]=statKalmanFilterFast(Y,A,C,Q,R,x0,P0,B,D,U,outRejFlag);
else
  %[Xf,Pf,Xp,Pp,rejSamples]=filterStationary_wConstraint(Y,A,C,Q,R,x0,P0,B,D,U,constFun); 
end

%Step 2: backward pass: (following the Rauch-Tung-Striebel implementation:
%https://en.wikipedia.org/wiki/Kalman_filter#Fixed-interval_smoothers)
Xs=Xf;
Ps=Pf;
prevXs=Xf(:,end);
prevPs=Pf(:,:,end);
%S=pinv(Q)*A;
D1=size(A,1);
if isa(Xs,'gpuArray') %For code to work on gpu
    Pt=nan(D1,D1,size(Y,2)-1,'gpuArray'); %Transition covariance matrix
else
    Pt=nan(D1,D1,size(Y,2)-1); %Transition covariance matrix
end

Mm=3;
for i=(size(Y,2)-1):-1:(size(Y,2)-Mm)
  
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
  newPt=prevPs*newK';
  newPs=pf+newK*(newPt-AP); %=newK*prevPs'*newK' + pf -pf'*(A'/pp)*A*pf =  newK*prevPs'*newK' + pf -((pf'*A')/(A*pf*A'+Q))*A*pf = 
  newPs=(newPs+newPs')/2; %Ugly hack to ensure symmetry of matrix
  
  %xp=A*xf+B*U(:,i); %Could compute instead of acccesing it, unclear which is faster
  prevXs=xf + newK*(prevXs-xp); 
  
  Xs(:,i)=prevXs;
  Pt(:,:,i)=newPt;
  
  %Improved (smoothed) covariance estimate
  prevPs=newPs;
  Ps(:,:,i)=prevPs;
end
%From now on, assume steady-state:
Pt(:,:,1:(size(Y,2)-Mm))=repmat(newPt,1,1,(size(Y,2)-Mm));
Ps(:,:,1:(size(Y,2)-Mm))=repmat(newPs,1,1,(size(Y,2)-Mm));
aux=Xf-newK*Xp(:,2:end);
for i=(size(Y,2)-Mm-1):-1:1
    %xp=Xp(:,i+1);
    %xf=Xf(:,i);
    %prevXs=xf + newK*prevXs-newK*xp; 
    prevXs=aux(:,i) + newK*prevXs;
    Xs(:,i)=prevXs;
end

end
