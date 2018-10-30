function [Xs,Ps,Pt,Xf,Pf,Xp,Pp,rejSamples]=statKalmanSmoother(Y,A,C,Q,R,x0,P0,B,D,U,outRejFlag,fastFlag,Ub)
%Implements a Kalman smoother for a stationary system
%INPUT:
%Y: D1xN observed data
%U: D3xN input data
%A,C,Q,R,B,D: system parameters, B,D,U are optional (default=0)
%x0,P0: initial guess of state and covariance, optional
%outRejFlag: flag to indicate if outlier rejection should be performed
%fastFlag: flag to indicate if fast smoothing should be performed. Default is no. Empty flag means no, any other value is yes.
%OUTPUT:
%Xs: D1xN, MLE estimate of state after smoothing
%Ps: D1xD1xN, covariance of state after smoothing
%Pt: D1xD1x(N-1) covariance of state transitions after smoothing
%Xf: D1xN, MLE estimate of state after FILTERING (i.e. forward pass only), see statKalmanFilter()
%Pf: D1xD1xN, covariance of state after FILTERING (i.e. forward pass only), see statKalmanFilter()
%See also:
% statKalmanFilter, filterStationary_wConstraint, EM

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
  outRejFlag=0; %No outlier rejection
end
if nargin<12 || isempty(fastFlag) || fastFlag==0 || fastFlag>=(N-1)
    M=N-1; %Do true filtering for all samples
elseif fastFlag==1
    M2=20; %Default for fast filtering: 20 samples
    M1=ceil(3*max(-1./log(abs(eig(A))))); %This many strides ensures ~convergence of gains before we assume steady-state
    M=max(M1,M2);
    M=min(M,N-1); %Prevent more than N-1, if this happens, we are not doing fast filtering
else
    M=min(ceil(abs(fastFlag)),N-1); %If fastFlag is a number but not 0 or 1, use that as number of samples
    M1=ceil(3*max(-1./log(abs(eig(A))))); %This many strides ensures ~convergence of gains before we assume steady-state
    if M<(N-1) && M<M1 %If number of samples provided is not ALL of them, but eigenvalues suggest the system is slower than provided number
        warning('statKSfast:fewSamples','Number of samples for fast filtering were provided, but system time-constants indicate more are needed')
    end
end
Ud=U;
if nargin<13 %Allowing for different inputs to output and dynamics equations
  Ub=U;
end

if M<(N-1) && any(abs(eig(A))>1)
    %If the system is unstable, there is no guarantee that the kalman gain
    %converges, and the fast filtering will lead to divergence of estimates
    warning('statKSfast:unstable','Doing steady-state (fast) filtering on an unstable system. States will diverge. Doing traditional filtering instead.')
    M=N-1;
end

%Size checks:
%TODO

%Step 1: forward filter
[Xf,Pf,Xp,Pp,rejSamples]=statKalmanFilter(Y,A,C,Q,R,x0,P0,B,D,Ud,outRejFlag,M+1,Ub);

%Step 2: backward pass: (following the Rauch-Tung-Striebel implementation:
%https://en.wikipedia.org/wiki/Kalman_filter#Fixed-interval_smoothers)

%Special case: deterministic system, no filtering needed. This can also be the case if Q << C'*R*C, and the system is stable
% if all(Q(:)==0)
%     Xs=Xf;
%     Ps=Pf;
%     Pt=Ps;
%     return
% end

D1=size(A,1);

%Initialize last sample:
Xs=nan(size(Xf));
Ps=nan(size(Pf));
prevPs=Pf(:,:,N);
prevXs=Xf(:,N);
Ps(:,:,N)=prevPs;
Xs(:,N)=prevXs;

if isa(Xs,'gpuArray') %For code to work on gpu
    Pt=nan(D1,D1,N-1,'gpuArray'); %Transition covariance matrix
else
    Pt=nan(D1,D1,N-1); %Transition covariance matrix
end

%Separate samples into fast and normal filtering intervals:
M1=M;
M2=M;
Nfast=N-1-(M1+M2);
if Nfast<0 %No fast filtering at all
    M1=N-1;
    M2=0;
    Nfast=0;
end

%Do true smoothing for last M1 samples:
for i=N-1:-1:N-M1
  %First, get estimates from forward pass:
  xf=Xf(:,i); %Previous posterior estimate of covariance at this step
  pf=Pf(:,:,i); %Previous posterior estimate of covariance at this time step
  xp=Xp(:,i+1); %Prediction of next step based on post estimate of this step
  pp=Pp(:,:,i+1); %Covariance of next step based on post estimate of this step

  %Backward pass:
  [prevPs,prevXs,newPt]=backStep(pp,pf,prevPs,xp,xf,prevXs,A);

  %Store estimates:
  Xs(:,i)=prevXs;
  Pt(:,:,i)=newPt;
  Ps(:,:,i)=prevPs;
end

%Fast smoothing for the middle (N-2*M) samples
if Nfast>0 %Assume steady-state:
    [icP,~]=pinvchol(pp);
    H=(pf*(A'*icP))*icP'; %TODO: check for stability efficiently
     if any(abs(eig(H))>1)
         warning('statKS:unstableSmooth','Unstable smoothing, skipping the backward pass.')
         H=zeros(size(H));
     end
    aux=Xf-H*Xp(:,2:end); %Precompute for speed
    for i=(N-M1-1):-1:(M2+1)
        prevXs=aux(:,i) + H*prevXs; %=Xf(:,i) + H*(prevXs-Xp(:,i+1));
        Xs(:,i)=prevXs;
    end
    %Compute covariances if requested:
    if nargout>2
        Ps(:,:,M2+1:N-M1-1)=repmat(prevPs,1,1,Nfast);
    end
    if nargout>3
        Pt(:,:,M2+1:N-M1-1)=repmat(newPt,1,1,Nfast);
    end
end

%Do true smoothing for first M2 samples:
for i=M2:-1:1
  %First, get estimates from forward pass:
  xf=Xf(:,i); %Previous posterior estimate of covariance at this step
  pf=Pf(:,:,i); %Previous posterior estimate of covariance at this time step
  xp=Xp(:,i+1); %Prediction of next step based on post estimate of this step
  pp=Pp(:,:,i+1); %Covariance of next step based on post estimate of this step

  %Backward pass:
  [prevPs,prevXs,newPt]=backStep(pp,pf,prevPs,xp,xf,prevXs,A);

  %Store estimates:
  Xs(:,i)=prevXs;
  Pt(:,:,i)=newPt;
  Ps(:,:,i)=prevPs;
end

end

function [newPs,newXs,newPt]=backStep(pp,pf,ps,xp,xf,prevXs,A)
  %Implements the Rauch-Tung-Striebel backward recursion
  %First, compute gain:
  [icP,~]=pinvchol(pp);
  H=(pf*(A'*icP))*icP'; %H=AP'/pp; %Faster, although worse conditioned, matters a lot when smoothing

  %Unstable smoothing warning: in general, the back step can be unstable for a few strides, but if it happens always, there is probably something wrong:
  %if any(abs(eig(H))>1)
  %  warning('Unstable back filter!')
  %end

  %Compute relevant covariances
  newPt=ps*H'; %This should be such that A*newPt' is hermitian
  Hext=H*(mycholcov(pp-ps)');

  %Updates: Improved (smoothed) state estimate
  newPs=pf-Hext*Hext'; %=pf + Hps*Hps' - Hcpp'*Hcpp;  %Would it be more precise/efficient to compute the sum of the last two terms as inv(inv(pf) +A'*inv(Q)*A) ?
  cPs=mycholcov(newPs); %Ensure PSD
  newPs=cPs'*cPs;
  newXs=xf + H*(prevXs-xp);
end

function [newPs,newXs,newPt,newDelta,newLambda]=backStep2(xf,Pf,CtRinvC,I_KC,prevDelta,prevLambda,CtRinv)
  %Implements the modified Bryson-Frazier smoother, which does not invert the covariances
  newDelta=A'*(CtRinvC +I_KC'*prevDelta*I_KC)*A;
  newLambda=A'*(I_KC*prevLambda - CtRinv*innov);
  newXs=xf-Pf*newLambda;
  newPs=Pf-Pf*newDelta*Pf;
  newPt=[]; %??
end
