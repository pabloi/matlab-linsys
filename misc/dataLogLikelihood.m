function logLperSamplePerDim=dataLogLikelihood(Y,U,A,B,C,D,Q,R,X0,P0,method)
%Evaluates the likelihood of the data under a given model

if nargin<11 || isempty(method)
    method='approx'; %Computing approx version by default, exact is too slow
end
if nargin<10 || isempty(X0) || isempty(P0)
    X0=[];
    P0=[];
end

if isa(Y,'cell') %Case where input/output data corresponds to many realizations of system, requires Y,U,x0,P0 to be cells of same size
    logLperSamplePerDim=cellfun(@(y,u,x0,p0) dataLogLikelihood(y,u,A,B,C,D,Q,R,x0,p0,method),Y,U,X0,P0);
    sampleSize=cellfun(@(y) size(y,2),Y);
    logLperSamplePerDim=(logLperSamplePerDim*sampleSize')/sum(sampleSize);
else

    if size(X0,2)<=1 %True init state guess
        fastFlag=[]; %Traditional (slow) filter, set to 0 for fast filtering
        [~,~,Xp,Pp,~]=statKalmanFilter(Y,A,C,Q,R,X0,P0,B,D,U,[],fastFlag);
    else %whole filtered priors are provided, not just t=0
        Xp=X0;
        Pp=P0;
    end

    %'Incomplete' logLikelihood: p({y}|params) [Albert and Shadmehr 2017, eq. A1.25]
    predY=C*Xp(:,1:end-1)+D*U;
    z=Y-predY; %If this values are too high, it may be convenient to just set logL=-Inf, for numerical reasons
    idx=~any(isnan(Y));
    z=z(:,idx);

    switch method
        case 'approx'
            logLperSamplePerDim=logLapprox(z,Pp,C,R);
        case 'exact'
            logLperSamplePerDim=logLexact(z,Pp,C,R);
        case 'max'
            logLperSamplePerDim=logLopt(z);
        case 'fast'
            logLperSamplePerDim=logLfast(z,Pp,C,R,A);
    end

end
end

function logLperSamplePerDim=logLexact(z,Pp,C,R)
[D2,N2]=size(z);
%Exact way: (very slow)
minus2ly=nan(size(z,2),1);
for i=1:size(z,2)
    sP=mycholcov(Pp(:,:,i));
    CsP=C*sP';
    P=R+CsP*CsP';
    [cP,r]=chol(P); %This has to be PD strictly
    logdetP= 2*sum(log(diag(cP)));
    %logdetP= sum(log(eig(P)));%Should use:https://en.wikipedia.org/wiki/Matrix_determinant_lemma to cheapen computation (can exploit knowing C'*(R\C) and det(R) ahead of time to only need computing size(Pp) determinants
    %eP=eig(P);
    %if ~all(imag(eP)==0 & eP>0) %Sanity check, the output covariance should be positive semidef., otherwise the likelihood is not well defined
    if r~=0
        error('Covariance matrix is not PD, cannot compute likelihood')
    end
    zz=z(:,i);
    minus2ly(i)=zz'*(P\zz) +logdetP + D2*log(2*pi);
end
logL=-.5*(sum(minus2ly));
logLperSamplePerDim=logL/(N2*D2);
end

function logLperSamplePerDim=logLapprox(z,Pp,C,R)
%Faster, approximate way:
[D,N]=size(z);
mPP=median(Pp,3);
cPP=mycholcov(mPP);
CcPP=C*cPP;
%CPpCt=C*median(Pp,3)*C'; %Mean makes no sense, changed to median
%CPpCt=(CPpCt+CPpCt')/2; %Cheap way to ensure PSD
CPpCt=CcPP*CcPP';
P=R+CPpCt;
[cP,r]=chol(P); %This has to be PD strictly
%eP=eig(P);
%if ~all(imag(eP)==0 & eP>0) %Sanity check, the output covariance should be positive semidef., otherwise the likelihood is not well defined
if r~=0
    error('Covariance matrix is not PD, cannot compute likelihood')
end
%logdetP= mean(log(eP)); %Should use:https://en.wikipedia.org/wiki/Matrix_determinant_lemma to cheapen computation (can exploit knowing C'*(R\C) and det(R) ahead of time to only need computing size(Pp) determinants
logdetP= 2*mean(log(diag(cP)));
S=z*z'/N;
%logL=-.5*N2*(trace(lsqminnorm(P,S,1e-8))+logdetP+D2*log(2*pi)); %Non-gpu ready
logLperSamplePerDim=-.5*(mean(diag(P\S))+logdetP+log(2*pi));
%Naturally, this is maximized over positive semidef. P (for a given set of residuals z)
%when P=S, and then it only depends on the sample covariance of the residuals:
%maxLperSamplePerDim = -.5*(1+mean(log(eig(S)))+log(2*pi))
end

function logLperSamplePerDim=logLfast(z,Pp,C,R,A)
%Do exact for M samples, do approximate afterwards. Should be almost equal
%to the exact version, but much faster. We exploit the fact that on a
%stable system the uncertainty reaches a steady-state, so the computation
%reduces to that of the approximate version

[D,N]=size(z);
M2=20; %Default for fast filtering: 20 samples
M1=ceil(3*max(-1./log(abs(eig(A))))); %This many strides ensures ~convergence of gains before we assume steady-state
M=max(M1,M2);
M=min(M,N); %Prevent more than N, if this happens, we are not doing fast filtering

logLperSamplePerDim1=logLexact(z(:,1:M),Pp(:,:,1:M),C,R);
logLperSamplePerDim2=logLapprox(z(:,M+1:N),Pp(:,:,M+1:N),C,R);
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
