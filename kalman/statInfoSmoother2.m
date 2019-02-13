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

[D2,N]=size(Y); D1=size(A,1);


%Init missing params:
aux=varargin;
[x0,P0,B,D,U,opts]=processKalmanOpts(D1,N,aux);
M=processFastFlag(opts.fastFlag,A,N);
opts.fastFlag=M+1;

%Size checks:
%TODO
rejSamples=[];
logL=[];

%Step 1: forward filter (could also be achieved through KF, but this
%precomputes the information matrices that are needed for the merge step)
%[Xf,Pf,Xp,Pp,rejSamples,logL]=statKalmanFilter(Y,A,C,Q,R,x0,P0,B,D,U,opts);
%[Xf,Pf,Xp,Pp,rejSamples,logL,Ip,~]=statInfoFilter(Y,A,C,Q,R,x0,P0,B,D,U,opts);
[iif,If,ip,Ip]=statInfoFilter2(Y,A,C,Q,R,x0,P0,B,D,U,opts);
is=zeros(size(iif));
Is=zeros(size(If));
if nargout>6
    Ps=zeros(size(Is));
    PtAt=zeros(size(Is));
    xs=zeros(size(is));
end
%Step 2: backward pass: (could also be achieved with a KF)
iA=inv(A);
%[Xfb,Pfb]=statKalmanFilter(fliplr(Y),iA,C,Q,R,x0,[],-iA*B,D,fliplr(U),opts);
%[Xfb,~,~,~,~,~,~,Ib]=statInfoFilter(fliplr(Y),iA,C,Q,R,x0,P0,-iA*B,D,fliplr(U),opts);
[ifb,Ib]=statInfoFilter2(fliplr(Y),iA,C,Q,R,x0,P0,-iA*B,D,fliplr(U),opts);

%Step 3: merge
for i=1:N
    iP1=Ib(:,:,N-i+1);%inv(P1);
    iP2=Ip(:,:,i);%inv(P2);
    I=(iP1+iP2);
    Is(:,:,i)=I;
    ii=ifb(:,N-i+1)+ip(:,i);
    is(:,i)=ii;
    if nargout>6
        [~,~,P]=pinvchol(I);
        xs(:,i)=P*ii;
        Ps(:,:,i)=P;
        PtAt(:,:,i)=P-P*I*Q; %Pt*A' = Ps*Ip*A*Pf*A' = Ps*Ip*(Pp-Q);
    end
end
end
