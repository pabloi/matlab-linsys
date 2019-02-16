function [ii,I,ip,Ip,x,P]=statInfoFilter2(Y,A,C,Q,R,varargin)
%statInfoFilter implements the Information formulation of the Kalman filter
%assuming stationary (fixed) noise matrices and system dynamics
%The model is: x[k+1]=A*x[k]+b+v[k], v~N(0,Q)
%y[k]=C*x[k]+d+w[k], w~N(0,R)
%This is a wrapped for trueStatInfoFilter, converting states to info and back,
%so that it has the same signature as statKalmanFilter
%INPUTS:
%
%OUTPUTS:
%
%See also: statInfoSmoother, statKalmanFilter, infoUpdate, KFupdate


[D2,N]=size(Y); D1=size(A,1);
%Init missing params:
[x0,P0,B,D,U,opts]=processKalmanOpts(D1,N,varargin);
M=processFastFlag(opts.fastFlag,A,N);

%TODO: Special case: deterministic system, no filtering needed. This can also be
%the case if Q << C'*R*C, and the system is stable

%Size checks:
%TODO

%Re-define observations to account for input effect:
Y_D=Y-D*U; BU=B*U;

%Define constants for sample rejection:
logL=nan(1,N); %Row vector
if opts.outlierFlag
    warning('Sample rejection not implemented for information filter')
end

%Precompute for efficiency:
[CtRinvC,~,CtRinvY]=reduceModel(C,R,Y_D);

%Convert init state to init info
[previ,prevI]=state2info(x0,P0);

%Run filter:
if M<N %Do the fast filtering for any remaining steps:
    warning('Fast mode not implemented in information filter')
    M=N;
end
[ii,I,ip,Ip]=trueStatInfoFilter(CtRinvY,CtRinvC,A,Q,BU,previ,prevI);

%To DO: if user requested output states and variances, compute them
if nargout>4
    for i=1:N
        [x(:,i),P(:,:,i)]=info2state(ii(:,i),I(:,:,i));
    end
end
