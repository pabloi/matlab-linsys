function logLperSamplePerDim=logLincomplete(Y,U,A,B,C,D,Q,R,X0,P0,method)
%Evaluates the likelihood of the data under a given model

if nargin<11 || isempty(method)
    method='approx'; %Computing approx version by default, exact is too slow
end
if nargin<10 || isempty(X0) || isempty(P0)
    warning('logLincomplete:noPriorGiven','No prior was provided to logLincomplete. Assuming an uninformative prior.')
    X0=zeros(size(A,1),1);
    P0=1e8*zeros(size(A,1));
    %Assuming a prior will create issues: we are essentially setting the first value of the likelihood factorization to be very close to 0. Even when taking the log(), this means subtracting a large term. If comparisons are made under the same conditions, this may be fine. Also, the 'approx' version of the logL can be immune to this issue, as it doesn't use the actual uncertainty but the MEDIAN uncertainty (i.e. we treat the uncertainty as if it had reached its steady-state, even for the very first stride.). If the inital state guess is good enough, then there is no large term subtraction.
    %TO DO: set X0,P0 to either the kalman smoother estimate (which is the MLE given everything else) OR to at least the kalman filtered value for it. In either case we have a bad init guess and we use some info to make a better one.
end

%Check if R is psd, and do cholesky decomp:
[cR,r]=mycholcov(R);
if r~=size(R,1)
    warning('R is not PD, this can end badly')
    %This is ok as long as R +C*P*C' is strictly positive definite.
    %Otherwise, nothing good can happen here, unless magically the residuals z have no projection on the null-space of R+C*P*C', and then we *could* compute a pseudo-likelihood by reducing the output to exclude the projection onto the null space, and reducing R+C*P*C' accordingly. However, that changes the dimension of the problem.
end

if isa(Y,'cell') %Case where input/output data corresponds to many realizations of system, requires Y,U,x0,P0 to be cells of same size
    logLperSamplePerDim=cellfun(@(y,u,x0,p0) logLincomplete(y,u,A,B,C,D,Q,R,x0,p0,method),Y,U,X0,P0);
    sampleSize=cellfun(@(y) size(y,2),Y);
    logLperSamplePerDim=(logLperSamplePerDim*sampleSize')/sum(sampleSize);
else
    if size(X0,2)==1 %Only the initial guess was given, as should be
        fastFlag=false;
        [~,~,Xp,Pp,~]=statKalmanFilter(Y,A,C,Q,R,X0(:,1),P0(:,:,1),B,D,U,[],fastFlag);
    else %Allowing for the pre-computation of the filtered states. This is useful in the EM iteration to avoid doing the filtering twice in each Step
        %To do: check that matrices are of appropriate sizes: X0 is nd x (N+1) and P0 is nd x nd x (N+1)
        Xp=X0;
        Pp=P0;
    end

    %'Incomplete' logLikelihood: p({y}|params) [Albert and Shadmehr 2017, eq. A1.25]
    predY=C*Xp(:,1:end-1)+D*U; %Prediction from one step before
    z=Y-predY; %If this values are too high, it may be convenient to just set logL=-Inf, for numerical reasons
    idx=~any(isnan(Y));
    z=z(:,idx); %Removing NaN samples

    switch method
        case 'approx'
            logLperSamplePerDim=logLapprox(z,Pp,C,cR);
        case 'exact'
            logLperSamplePerDim=logLexact(z,Pp,C,cR);
        case 'max'
            logLperSamplePerDim=logLopt(z);
        case 'fast'
            logLperSamplePerDim=logLfast(z,Pp,C,cR,A);
    end

end
end

function logLperSamplePerDim=logLexact(z,Pp,C,cR)
[D2,N2]=size(z);
%Exact way: (10x slower than the approximate way)
minus2ly=nan(size(z,2),1);
for i=1:size(z,2)
    [cP]=RplusCPC(cR,Pp(:,:,i),C);
    logdetP= 2*sum(log(diag(cP)));
    zz=z(:,i);  Pz=cP'\zz;
    minus2ly(i)=Pz'*Pz + logdetP + D2*log(2*pi);
end
logLperSamplePerDim=-.5*(sum(minus2ly))/(N2*D2);
end

function logLperSamplePerDim=logLapprox(z,Pp,C,cR)
%Faster, approximate way: essentially we assume that the uncertainty is in a steady-state throughout all the data, and use the median uncertainty as a proxy for the steady-state value.
[D,N]=size(z);
mPP=median(Pp,3);
[cP,r]=RplusCPC(cR,mPP,C);
logdetP= 2*mean(log(diag(cP)));
%S=z*z'/N;
%logLperSamplePerDim=-.5*(mean(diag(P\S))+logdetP+log(2*pi));
Pz=cP'\z;
logLperSamplePerDim=-.5*(mean(mean(Pz.*Pz))+logdetP+log(2*pi));
%Naturally, this is maximized over positive semidef. P (for a given set of residuals z)
%when P=S, and then it only depends on the sample covariance of the residuals:
%maxLperSamplePerDim = -.5*(1+mean(log(eig(S)))+log(2*pi))
end

function logLperSamplePerDim=logLfast(z,Pp,C,cR,A)
%Do exact for M samples, do approximate afterwards. Should be almost equal
%to the exact version, but much faster. We exploit the fact that on a
%stable system the uncertainty reaches a steady-state, so the computation
%reduces to that of the approximate version

[D,N]=size(z);
M2=20; %Default for fast filtering: 20 samples
M1=ceil(3*max(-1./log(abs(eig(A))))); %This many strides ensures ~convergence of gains before we assume steady-state
M=max(M1,M2);
M=min(M,N); %Prevent more than N, if this happens, we are not doing fast filtering

logLperSamplePerDim1=logLexact(z(:,1:M),Pp(:,:,1:M),C,cR);
logLperSamplePerDim2=logLapprox(z(:,M+1:N),Pp(:,:,M+1:N),C,cR);
logLperSamplePerDim=(M*logLperSamplePerDim1+(N-M)*logLperSamplePerDim2)/N;

end

function logLperSamplePerDim=logLopt(z)
%Max possible log-likelihood over all matrices P
%See refineQR for computing optimal values of Q,R that preserve z and
%maximize logL
[D2,N2]=size(z);
u=z/sqrt(N2);
S=u*u';
logdetS=mean(log(eig(S)));
logLperSamplePerDim = -.5*(1+log(2*pi)+logdetS);
%Special case: if S is isotropic (all eigenvalues are the same), then log(det(S))=D2*log(trace(S)/D2) and N2*trace(S)=norm(z,'fro')^2
%Thus: logL = -.5*N2*D2*(1+log(norm(z,'fro')^2/N2)-log(D2)+log(2*pi))
%logL=N2*maxLperSample;
end

function [cP,P]=RplusCPC(cR,P,C)
    %Summing in chol() space to guarantee that x'*P*x products are non-negative.
    cP1=mycholcov(P); %This can be PSD
    %Option 1: %The most accurate as far as I can tell, but not the fastest.
    CcP=C*cP1';  P=cR'*cR+CcP*CcP'; [cP,r]=mycholcov(P); %This HAS TO BE PD. If not, best case is numerical error that makes a PD matrix look like indefinite.
    if r<size(C,1)
      warning('dataLogL:nonPDcov','R+C*P*C^t was not positive definite. LogL is not defined. Regularizing to move forward.')
      cP=[cP;zeros(size(C,1)-r,size(C,1))];
      cP=cP+1e-11*eye(size(C,1));
    end

    %Option 2: %Slightly faster, as it exploits the Cholesky decomp. Less accurate in general though, see testPDSsum
    %(x'*P*x has an error of about an order of magnitude larger. However, the typical error is around 1e-29, see testPDSsum).
    %[v,d]=eig(P);    %d=sqrt(d); % [cP]=myPSDsum(cR,[],C*v*d);

    %Option 3: %Alternative to above using svd() instead of eig(), which is arguably more accurate.
    %[~,d,v]=svd(cP);
    %[cP]=myPSDsum(cR,[],C*v*d); %To Do: check which is larger of Pp,R, and do the update in the more convenient form
end
