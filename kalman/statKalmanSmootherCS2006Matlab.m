function [Xs,Ps,Pt,logL]=statKalmanSmootherCS2006Matlab(Y,A,C,Q,R,varargin)
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

% Reduce model
Y_D=Y-D*U;
logLmargin=0;
if ~opts.noReduceFlag && D2>D1 %Reducing model if useful
[CtRinvC,~,CtRinvY,~,logLmargin]=reduceModel(C,R,Y_D);
C=CtRinvC; R=CtRinvC; Y_D=CtRinvY; %Reducing dimension of problem for speed
D=zeros(D1,size(U,1));
end

%For the first steps do an information update if P0 contains infinite elements
%This is a doxy, because the C code does not handle infinite covariances
%That problem can be compounded if there are infinite covariances AND
%missing data for the first sample (e.g. during cross-validation)
%firstInd=1;
prevP=P0;
prevX=x0;
%For the first steps do an information update if P0 contains infinite elements
firstInd=1;
infVariances=isinf(diag(prevP));
while any(infVariances)  %My filter uses Inf as initial uncertainty, but CS2006 does not support it, or anything too large
    prevP=1e9*eye(size(prevP)); %Workaround
    prevX=zeros(size(prevX));
end

%Now do the actual thing:
LDS.A=A;
LDS.B=B;
LDS.C=C;
LDS.D=D;
LDS.Q=Q;
LDS.R=R;
LDS.x0=prevX;
LDS.V0=prevP; %My filter uses Inf as initial uncertainty, but CS2006 does not support it, or anything too large
%if any(isnan(Y_D(:,1)))
%    Y_D(:,1)=0;
%end
[Lik,Xs,Ps,Pt] = SmoothLDSMatlab(LDS,Y_D,U,U); %Mex version (requires building from mex, see README in lds-1.1)
aux=Lik+logLmargin;
logL=nanmean(aux(firstInd:end))/size(Y,1); %Per-sample, per-dimension of output
end
