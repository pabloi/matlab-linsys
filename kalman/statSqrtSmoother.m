function [Xs,Ps,Pt,Xf,Pf,rejSamples,logL]=statSqrtSmoother(Y,A,C,Q,R,varargin)
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
[Xf,Pf,rejSamples,logL]=statSqrtFilter(Y,A,C,Q,R,x0,P0,B,D,U,opts);


%Step 2: backward pass:
%TODO: Special case: deterministic system, no filtering needed. This can also be the case if Q << C'*R*C, and the system is stable

D1=size(A,1);
%Initialize last sample:
Xs=nan(size(Xf)); prevXs=Xf(:,N);   Xs(:,N)=prevXs;
Ps=nan(size(Pf)); prevPs=Pf(:,:,N); Ps(:,:,N)=prevPs;

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

%iA=inv(A); %If this throws a warning A may not be invertible
cQt=mycholcov2(Q)';
At=A';
for i=N-1:-1:N-M1
  %First, get estimates from forward pass:
  xf=Xf(:,i); %Previous posterior estimate of covariance at this step
  pf=Pf(:,:,i); %Previous posterior estimate of covariance at this time step
  bu=BU(:,i);
  xp=A*xf+bu;

  %Backward pass:
  [prevPs,prevXs,newPt,Ht]=sqrtRTS(pf,prevPs,xp,xf,prevXs,At,cQt);

  %Store estimates:
  Xs(:,i)=prevXs;  Pt(:,:,i)=newPt;  Ps(:,:,i)=prevPs;
end

%Fast smoothing for the middle (N-2*M) samples (using the
%Rauch-Tung-Striebel equations, should see how to do the BF equations)
if Nfast>0 %Assume steady-state:
    H=Ht';
     if any(abs(eig(H))>1) %TODO: check for stability efficiently
         warning('statKS:unstableSmooth','Unstable smoothing, skipping the backward pass.')
         H=zeros(size(H));
     end
    aux=Xf-H*(A*Xf+BU); %Precompute for speed
    for i=(N-M1-1):-1:(M2+1)
        prevXs=aux(:,i) + H*prevXs; %=Xf(:,i) + H*(prevXs-Xp(:,i+1));
        Xs(:,i)=prevXs;
    end
    %Compute covariances if requested:
    if nargout>2; Ps(:,:,M2+1:N-M1-1)=repmat(prevPs,1,1,Nfast);
        if nargout>3; Pt(:,:,M2+1:N-M1-1)=repmat(newPt,1,1,Nfast); end
    end
end

%Do true smoothing for first M2 samples:
for i=M2:-1:1
  %First, get estimates from forward pass:
  xf=Xf(:,i); %Previous posterior estimate of covariance at this step
  pf=Pf(:,:,i); %Previous posterior estimate of covariance at this time step
  bu=BU(:,i);
  xp=A*xf+bu;

  %Backward pass:
  [prevPs,prevXs,newPt,Ht]=sqrtRTS(pf,prevPs,xp,xf,prevXs,At,cQt);

  %Store estimates:
  Xs(:,i)=prevXs;  Pt(:,:,i)=newPt;  Ps(:,:,i)=prevPs;
end

end