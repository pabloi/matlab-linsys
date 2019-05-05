function [Xs,Ps,Pt,Xf,Pf,Xp,Pp,rejSamples,logL]=statKalmanSmoother(Y,A,C,Q,R,varargin)
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
[Xf,Pf,Xp,Pp,rejSamples,logL]=statKalmanFilter(Y,A,C,Q,R,x0,P0,B,D,U,opts);

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

%Do true smoothing for last M1 samples:
%if D2>D1 && ~opts.noReduceFlag %Reducing dimension of problem for speed
%  [C,~,Y]=reduceModel(C,R,Y-D*U);
%end
%prevDelta=0; prevLambda=zeros(D1,1);
%Innov=Y-C*Xp(:,1:N);
pp=Pp(:,:,end);
iA=inv(A); %If this throws a warning A may not be invertible
cQ=mycholcov2(Q);
for i=N-1:-1:N-M1
  %First, get estimates from forward pass:
  xf=Xf(:,i); %Previous posterior estimate of covariance at this step
  pf=Pf(:,:,i); %Previous posterior estimate of covariance at this time step
  xp=Xp(:,i+1); %Prediction of next step based on post estimate of this step
  %prevpp=pp;
  pp=Pp(:,:,i+1); %Covariance of next step based on post estimate of this step
  bu=BU(:,i);

  %Backward pass:
  [prevPs,prevXs,newPt]=backStepRTS(pp,pf,prevPs,xp,xf,prevXs,A,cQ,bu,iA);
  %Alt: Bryson-Frazier recursion. Faster, but error-prone
  %innov=Innov(:,i);
  %[prevPs,prevXs,newPt,prevDelta,prevLambda]=backStepBF(xf,pf,innov,pp,prevpp,A,C,invSchol(:,:,i),prevDelta,prevLambda);

  %Store estimates:
  Xs(:,i)=prevXs;  Pt(:,:,i)=newPt;  Ps(:,:,i)=prevPs;
end

%Fast smoothing for the middle (N-2*M) samples (using the
%Rauch-Tung-Striebel equations, should see how to do the BF equations)
if Nfast>0 %Assume steady-state:
    [~,~,newPt,H]=backStepRTS(pp,pf,prevPs,Xp(:,i+1),xf,prevXs,A,cQ,bu,iA); %Get gain H
     if any(abs(eig(H))>1) %TODO: check for stability efficiently
         warning('statKS:unstableSmooth','Unstable smoothing, skipping the backward pass.')
         H=zeros(size(H));
     end
    aux=Xf-H*Xp(:,2:end); %Precompute for speed
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
  xp=Xp(:,i+1); %Prediction of next step based on post estimate of this step
  %prevpp=pp;
  pp=Pp(:,:,i+1); %Covariance of next step based on post estimate of this step
  bu=BU(:,i);

  %Backward pass:
  [prevPs,prevXs,newPt]=backStepRTS(pp,pf,prevPs,xp,xf,prevXs,A,cQ,bu,iA);
  %Alt:
  %innov=Innov(:,i);
  %[prevPs,prevXs,newPt,prevDelta,prevLambda]=backStepBF(xf,pf,innov,pp,prevpp,A,C,invSchol(:,:,i),prevDelta,prevLambda);

  %Store estimates:
  Xs(:,i)=prevXs;  Pt(:,:,i)=newPt;  Ps(:,:,i)=prevPs;
end

end

function [newPs,newXs,newPt,H]=backStepRTS(pp,pf,ps,xp,xf,prevXs,A,cQ,bu,iA)
  %Implements the Rauch-Tung-Striebel backward recursion
  %https://en.wikipedia.org/wiki/Kalman_filter#Fixed-interval_smoothers)

  %Four cases to consider:
  %1) Pp has very large entries (possibly infinite): if A is invertible, use the alternate form, which is well conditioned.
  %2) Pp has very small ones (possibly 0): inverse of Pp does not exist. Bad idea. This means that two states are highly coupled or one state is known exactly (and therefore should not get updated on the smoothing pass). A possible strategy is to rotate the space such that one state is known exactly and try again with the remaining states. Another strategy is to inject artificial uncertainty.
  %3) Pp has both: As before, remove 0's and try again.
  %4) Pp has neither: standard recursion!
  cPs=mycholcov2(ps);
  infIdx=isinf(diag(pp));
  if ~any(infIdx) %The usual case: we have a proper/numerically well-conditioned prior from filter
      [icP,~]=pinvchol2(pp); %What happens if pp is NOT invertible (null eigenvalues)?
      %First, compute gain:
      HcP=(pf*(A'*icP)); %H*cP'
      H=HcP*icP'; %H=AP'/pp; %Faster, although worse conditioned, matters a lot when smoothing
      %Equivalent: (numerically too?)
      %H=pf*(A'*invPp);
      %HcP=H*cP';
      %State update:
      newXs=xf+H*(prevXs-xp); %=H*prevXs +(xf-H*xp); 
      %Compute across-steps covariance:
      newPt=ps*H'; %This should be such that A*newPt' is hermitian
      %More stable state covariance update:
      HcPs=H*cPs';
      newPs= HcPs*HcPs' + (pf - HcP*HcP'); %The term in parenthesis is psd although this is not numerically enforced
      %To enforce: (makes it slower)
      %cS=mycholcov(pf - HcP*HcP'); %Should be psd: proof? Can also be computed in a PSD-enforcing way by using chol(Pf). Solution: cS'*cS = cPf'*(eye - (cPf*A'*icP)*(cPf*A'*icP)')*cPf;
      %newPs=HcPs*HcPs'+cS'*cS;%=HcPs*HcPs' + (pf - HcP*HcP');%=pf- H*(pp-ps)*H'; %The term in parenthesis is psd = inv(inv(pf)+A'*inv(Q)*A)    
  else %This happens when we started filtering from an improper prior
    %(infinite uncertainty) or very large uncertainties and we go back to smooth
    %those (almost) infinitely uncertain samples
    %This is fine if A is invertible. Otherwise it will be problematic
    %(we cannot infer the previous state solely from the current one)
    [icP,~]=pinvchol2(pp); %Handles infinite (and 0 covariances) covariances. Substitutes non-diagonal Inf elements by 0.
    [newPs,newXs,newPt,H]=backStepRTS_invA(icP,cPs,xp,prevXs,cQ,bu,iA);
    %This function could be used whenever inv(A) is defined, not just with inifinte elements in Pp.
    %It may be better conditioned that standard RTS to deal with large Pp (trace(Pp)>> trace(Ps))
  end
end
function [newPs,newXs,newPt,H]=backStepRTS_invA(invCholPp,cholPs,xp,prevXs,cholQ,bu,iA)
  %An RTS-equivalent backward recursion assuming that A is invertible
  %Note that inverting Pp (as RTS requires), implies that A*Pf*A'+Q is invertible,
  %which in the general case requires invertible Pf and A, OR invertible Q.
  %Notably, this formulation does not require xf, Pf, or A, and can be understood as an
  %additive improvement on the estimate coming from (xs,Ps) of step k+1, whereas
  %the regular RTS is formulated as an additive improvement on (xf,Pf) form the
  %forward pass at step k.
  %This form makes it easy to update when Pp has infinite elements.
  %The main insight here is that if A is invertible, then:
  %H=inv(A)*(eye-Q*inv(pp));
  icP=invCholPp;
  %iP=icP*icP';
  iAcQ=iA*cholQ'; %Could precompute outside loop
  cQcP=cholQ*icP;
  F=iAcQ*cQcP*icP';
  H=iA-F;%=iA*(eye(size(Q))-Q*invPp); %Diagonal elements of Q*invPp have to be ALL less than 1
  HcPs=H*cholPs';
  newPt=cholPs'*HcPs';
  newPs= HcPs*HcPs' + iAcQ*(eye(size(iA))-cQcP*cQcP')*iAcQ'; %=H*Ps*H'+F*Pp*F'+iA*Q*iA';
  newXs=iA*(prevXs-bu) +F*(xp-prevXs);
end

function [newPs,newXs,newPt,newDelta,newLambda]=backStepBF(xf,Pf,innov,Pp,prevPp,A,C,icS,prevDelta,prevLambda)
  %Implements the modified Bryson-Frazier smoother, which does not invert the covariances
  %Using Martin 2014 equations, in which Pt is derived (wikipedia does not have it)
  %Wikipedia:
  %newDelta=A'*(CtRinvC +I_KC'*prevDelta*I_KC)*A;
  %newLambda=A'*(I_KC*prevLambda +CtRinvC*xp - CtRinvY);
  %newXs=xf-Pf*newLambda;
  %newPs=Pf-Pf*newDelta*Pf;
  %newPt=[]; %??
  %Martin 2014:
  PfA=Pf*A';
  newPs=Pf -PfA*prevDelta*PfA';
  newXs=xf - PfA*prevLambda;
  I=eye(size(Pp));
  newPt=(I-Pp*prevDelta)*PfA';
  CicS=C'*icS;
  CtinvSC=CicS*CicS';
  I_KC=I-prevPp*CtinvSC;
  I_KCA=I_KC'*A';
  newLambda=I_KCA*prevLambda - CicS*icS'*innov;
  newDelta=I_KCA*prevDelta*I_KCA' + CtinvSC;

end
