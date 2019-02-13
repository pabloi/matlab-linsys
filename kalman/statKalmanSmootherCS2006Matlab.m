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
% infVariances=isinf(diag(prevP));
% while any(infVariances) %In practice, this only gets executed until the first non-NaN data sample is found
%     %Define info matrix from cov matrix:
%     prevI=zeros(size(prevP));
%     aux=inv(prevP(~infVariances,~infVariances)); %This inverse needs to exist, no such thing as absolute certainty
%     prevI(~infVariances,~infVariances)=aux; %Computing inverse of the finite submatrix of P0
%     %prevI=diag(1./diag(P0)); %This information matrix ignores correlations, cheaper
%     %Update:
%     data=CtRinvY(:,firstInd);
%     if ~any(isnan(data)) %Do update
%         [~,~,prevX,prevP,logL(firstInd)]=infoUpdate(CtRinvC,data,prevX,prevP,prevI);
%         %Warning: if variance was inifinte, then logL(firstInd)=-Inf!
%         %Predict:
%         [prevX,prevP]=KFpredict(A,Q,prevX,prevP,BU(:,firstInd));
%     else %Sample was NaN
%         prevX=A*prevX+BU(:,firstInd); %Update the mean, even if uncertainty is infinite
%         prevP=A*prevP*A'+Q; %Propagating infinity
%     end
%     firstInd=firstInd+1;
%     %New variances:
%     infVariances=isinf(diag(prevP));
% end

%Now do the actual thing:
LDS.A=A;
LDS.B=B;
LDS.C=C;
LDS.D=D;
LDS.Q=Q;
LDS.R=R;
LDS.x0=prevX;
LDS.V0=prevP; %My filter uses Inf as initial uncertainty, but CS2006 does not support it, or anything too large
if any(isnan(Y_D(:,1)))
    Y_D(:,1)=0;
end
[Lik,Xs,Ps,Pt] = SmoothLDSMatlab(LDS,Y_D,U,U); %Mex version (requires building from mex, see README in lds-1.1)
aux=Lik*size(Y,2)/sum(~any(isnan(Y)))+nanmean(logLmargin);
logL=aux/size(Y,1); %Per-sample, per-dimension of output
end
