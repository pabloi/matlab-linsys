function [Xs,Ps,Pt,Xf,Pf,Xp,Pp,rejSamples,logL]=statInfoSmoother(Y,A,C,Q,R,varargin)
%Implements the Information smoother formulation of the Kalman smoother for a stationary system
%This implementation is faster (about 10%) than the classic Kalman smoother
%with reduced model when dim(y)>dim(x), and much faster than an
%implementation without the reduced model. The filter itself is slightly
%slower (if the log-likelihood is requested), but the smoothing can be 
%computed very efficiently with no matrix inversions required.
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
 
logL=NaN;
rejSamples=NaN;
[D2,N]=size(Y); D1=size(A,1);
%Init missing params:
[x0,P0,B,D,U,opts]=processKalmanOpts(D1,N,varargin);
[is,Is,iif,If,ip,Ip,Xs,Ps,Pt]=statInfoSmoother2(Y,A,C,Q,R,x0,P0,B,D,U,opts);
if nargout>3
    Pf=zeros(size(Is));
    Xf=zeros(size(is));
    for i=1:N
        I=If(:,:,i);
        [Xf(:,i),P]=info2state(iif(:,i),I);
        Pf(:,:,i)=P;
    end
end
if nargout>5
    Pp=zeros(size(Is));
    Xp=zeros(size(is));
    for i=1:N+1
        I=Ip(:,:,i);
        [Xp(:,i),P]=info2state(ip(:,i),I);
        Pp(:,:,i)=P;
    end
end