function [Xs,Ps,Pt,Xf,Pf,Xp,Pp,rejSamples,logL]=statKalmanSmoother2(Y,A,C,Q,R,varargin)
%Implements a Kalman smoother for a stationary system
%INPUT:
%Y: D1xN observed data
%U: D3xN input data
%A,C,Q,R,B,D: system parameters, B,D,U are optional (default=0)
%x0,P0: initial guess of state and covariance, optional
%outRejFlag: flag to indicate if outlier rejection should be performed
%fastFlag: flag to indicate if fast smoothing should be performed. Default is no. Empty flag or 0 means no, any other value is yes.
%OUTPUT:
%Xs: D1xN, MLE estimate of state after smoothing
%Ps: D1xD1xN, covariance of state after smoothing
%Pt: D1xD1x(N-1) covariance of state transitions after smoothing
%Xf: D1xN, MLE estimate of state after FILTERING (i.e. forward pass only), see statKalmanFilter()
%Pf: D1xD1xN, covariance of state after FILTERING (i.e. forward pass only), see statKalmanFilter()
%See also:
% statKalmanFilter, filterStationary_wConstraint, EM

[D2,N]=size(Y); D1=size(A,1);

%Init missing params:
aux=varargin;
[x0,P0,B,D,U,opts]=processKalmanOpts(D1,N,aux);
M=processFastFlag(opts.fastFlag,A,N);
opts.fastFlag=M+1;
BU=B*U;

%Size checks:
%TODO

%Step 1: forward filter
%[Xf,Pf,Xp,Pp,rejSamples,logL]=statKalmanFilter(Y,A,C,Q,R,x0,P0,B,D,U,opts);
[Xf,Pf,Xp,Pp,rejSamples,logL,S]=statKalmanFilter2(Y,A,C,Q,R,x0,P0,B,D,U,opts);
%Notice: Pp=VecU*Diag*VecU'; Pf=VecU*(Diag(-1)/(Diag(-1)+1)*VecU';

%Step 2: backward pass:

%TODO: Special case: deterministic system, no filtering needed. This can also be the case if Q << C'*R*C, and the system is stable

D1=size(A,1);
%Initialize last sample:
Xs=nan(size(Xf)); prevXs=Xf(:,N);   Xs(:,N)=prevXs;
Ps=nan(size(Pf)); prevPs=Ps(:,:,N); Ps(:,:,N)=prevPs;

if isa(Xs,'gpuArray') %For code to work on gpu
    Pt=nan(D1,D1,N-1,'gpuArray'); %Transition covariance matrix
else
    Pt=nan(D1,D1,N-1); %Transition covariance matrix
end

%Separate samples into fast and normal filtering intervals:
M1=M; M2=M; Nfast=N-1-(M1+M2);
if Nfast<=0 %No fast filtering at all
    M1=N-1; M2=0; Nfast=0;
end
prevPs=Pf(:,:,end)*Pf(:,:,end)';
for i=N-1:-1:1
  %First, get estimates from forward pass:
  xf=Xf(:,i); %Previous posterior estimate of covariance at this step
  xp=Xp(:,i+1); %Prediction of next step based on post estimate of this step
  icp=Pp(:,:,i+1); %sqrt inv matrix (chol-like)
  cpf=Pf(:,:,i); %sqrt of pf (chol-like)

  %Backward pass:
  %[prevPs,prevXs,newPt]=backStepRTS(pp,pf,prevPs,xp,xf,prevXs,A,cQ,bu,iA);
  [prevPs,prevXs,newPt]=backStepRTS(icp,cpf,prevPs,xp,xf,prevXs,A);
  
  %Store estimates:
  Xs(:,i)=prevXs;  Pt(:,:,i)=newPt;  Ps(:,:,i)=prevPs;
end

Xf=S*Xf;
Xp=S*Xp;
Xs=S*Xs;
%for i=1:N
%   P(:,:,i)=iS*P(:,:,i)*iS'; %PSD can be enforced by storing U and reconstructing P in a sqrt way
%   Pp(:,:,i)=iS*Pp(:,:,i)*iS';
%   Ps=
%   Pt=
%end

end

function [newPs,newXs,newPt]=backStepRTS(icP,cpf,ps,xp,xf,prevXs,A)
  %Implements the Rauch-Tung-Striebel backward recursion
  %https://en.wikipedia.org/wiki/Kalman_filter#Fixed-interval_smoothers)
  cPs=chol(ps); %This has minimal cost, as long as numerical issues do not keep ps from being PD
  %icP=U./d'; %Cholesky of inv(Pp)
  %First, compute gain:
  pf=cpf*cpf';
  HcP=pf*A'*icP;
  H=HcP*icP'; %H=AP'/pp; %Faster, although worse conditioned, matters a lot when smoothing
  %State update:
  newXs=xf+H*(prevXs-xp); %=H*prevXs +(xf-H*xp); 
  %Compute across-steps covariance:
  newPt=ps*H'; %This should be such that A*newPt' is hermitian
  %More stable state covariance update:
  HcPs=H*cPs';
  newPs= HcPs*HcPs' + (pf - HcP*HcP'); %The term in parenthesis is psd
  %although this is not numerically enforced, symmetry is enforced
end