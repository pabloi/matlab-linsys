function [is,Is,iif,If,ip,Ip,xs,Ps,PtAt]=statInfoSmoother2(Y,A,C,Q,R,varargin)
%Implements a Information-smoother for a stationary system, through
%independent forward and backward passes of an information filter, plus a
%subsequent merge phase.
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

%Init missing params:
aux=varargin;
D1=size(A,1);
[D2,N]=size(Y);
[x0,P0,B,D,U,opts]=processKalmanOpts(D1,N,aux);
M=processFastFlag(opts.fastFlag,A,N);

%Size checks:
%TODO
rejSamples=[];
logL=[];

%Re-define observations to account for input effect:
Y_D=Y-D*U; BU=B*U;

%Precompute for efficiency:
[CtRinvC,~,CtRinvY]=reduceModel(C,R,Y_D);
[iA]=pinv(A);
cQ=mycholcov(Q);
iAcQ=iA*cQ;

%Convert init state to init info
[previ,prevI]=state2info(x0,P0);


pp=gcp('nocreate');
if isempty(pp) %No parallel pool open, doing regular for, issue warning
    %warning('No parallel pool was found, running for loop in serial way.')
    %Step 1: forward filter
    [iif,If,ip,Ip]=trueStatInfoFilter(CtRinvY,CtRinvC,A,Q,BU,previ,prevI,M);

    %Step 2: backward pass: this is just running the filter backwards (makes sense only if A is invertible)
    %This can be run in parallel to the fwd filter!
    [ifb,Ib]=trueStatInfoFilter(fliplr(CtRinvY),CtRinvC,iA,iAcQ*iAcQ',-iA*fliplr([zeros(size(BU,1),1),BU(:,1:end-1)]),zeros(size(previ)),zeros(size(prevI)),M);
    %If the forward pass started from an uniformative prior (previ=0) then Ib
    %should be exactly If (minus numerical errors). If it started from
    %somewhere else, they should still converge to the same value.
    %Should I run the filter in both passes with uniformative priors, and an
    %extra time with the initial conditions to add the associated transient?

else %Running in parallel. Because of matlab's restrictions and overhead in parallelism, this is not worth it.
    error('Unimplemented parallel code')
    %args={CtRinvY,CtRinvC,A,Q,BU,previ,prevI;
    %    fliplr(CtRinvY),CtRinvC,iA,iAcQ*iAcQ',-iA*fliplr(BU),zeros(size(previ)),zeros(size(prevI))};
    %spmd
    %    if labIndex<2
    %        [i1,I1,i2,I2] = trueStatInfoFilter(args{labIndex,:});
    %    end
    %end
end

%Step 3: merge
Is=Ip(:,:,1:end-1)+flip(Ib,3);
is=ip(:,1:end-1)+fliplr(ifb);

if nargout>6
    Ps=zeros(size(Is));
    PtAt=zeros(size(Is));
    xs=zeros(size(is));
    for i=1:N
        I=Is(:,:,i);
        [xs(:,i),P]=info2state(is(:,i),I);
        Ps(:,:,i)=P;
        %PtAt(:,:,i)=P-P*Ip(:,:,i+1)*Q; %Pt*A' = Ps*Ip*A*Pf*A' = Ps*Ip*(Pp-Q) = Ps - Ps*Ip*Q;
        PtAt=[];
    end
end
