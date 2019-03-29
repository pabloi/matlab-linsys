function logL=dataLogLikelihood(Y,U,A,B,C,D,Q,R,X0,P0,method)
%Evaluates the likelihood of the data under a given model

if nargin<11 || isempty(method)
    method='exact';
end
if nargin<10 || isempty(X0) || isempty(P0)
    X0=[];
    P0=[];
end

%Check if R is psd, and do cholesky decomp:
[cR,r]=mycholcov(R);
if r~=size(R,1)
    warning('R is not PD, this can end badly')
    %This is ok as long as R +C*P*C' is strictly positive definite.
    %Otherwise, nothing good can happen here, unless magically the residuals z have no projection on the null-space of R+C*P*C', and then we *could* compute a pseudo-likelihood by reducing the output to exclude the projection onto the null space, and reducing R+C*P*C' accordingly. However, that changes the dimension of the problem.
end

if isa(Y,'cell') %Case where input/output data corresponds to many realizations of system, requires Y,U,x0,P0 to be cells of same size
    logL=cellfun(@(y,u,x0,p0) dataLogLikelihood(y,u,A,B,C,D,Q,R,x0,p0,method),Y,U,X0,P0);
    logL=sum(logL);
else

    if size(X0,2)<=1 %True init state guess
        opts.fastFlag=false;
        %Dalt=zeros(size(D,1),0);
        %opts1=opts;
        %opts1.indD=[];
        %[~,~,Xp,Pp,~,logL]=statKalmanFilter(Y-D*U,A,C,Q,R,X0,P0,B,Dalt,U,opts1);
        [~,~,Xp,Pp,~,logL]=statKalmanFilter(Y,A,C,Q,R,X0,P0,B,D,U,opts);
        if ~strcmp(method,'exact')
          warning('dataLogL:ignoreMethod','Method requested was not ''exact'', but returning exact log-likelihood anyway, because it''s faster.')
        end
        return; %statKF computes exact logL, so we are done
    else %whole filtered priors are provided, not just t=0
        Xp=X0;
        Pp=P0;
    end

    %'Incomplete' logLikelihood: p({y}|params) [Albert and Shadmehr 2017, eq. A1.25]
    predY=C*Xp(:,1:end-1)+D*U; %Prediction from one step ahead
    z=Y-predY; %If this values are too high, it may be convenient to just set logL=-Inf, for numerical reasons
    idx=~any(isnan(Y));
    z=z(:,idx); %Removing NaN samples
    Paux=Pp(:,:,1:end-1); %Uncertainty of one-step ahead state prediction
    Pp=Pp(:,:,idx); %Keeping only the ones not associated with NaN samples

    switch method
        case 'approx'
            logL=logLapprox(z,Pp,C,R);
        case 'exact'
            logL=logLexact(z,Pp,C,R);
        case 'max'
            logL=logLopt(z);
        case 'fast'
            logL=logLfast(z,Pp,C,R,A);
    end

end
end

function logL=logLexact(z,Pp,C,R)
[D,N]=size(z);
%Exact way: (10x slower than the approximate way)
minus2ly=nan(size(z,2),1);
for i=1:size(z,2)
    minus2ly(i)=logLnormal(z(:,i),R+C*Pp(:,:,i)*C');
end
logL=sum(minus2ly);
end

function logL=logLapprox(z,Pp,C,R)
%Faster, approximate way: essentially we assume that the uncertainty is in a steady-state throughout all the data, and use the median uncertainty as a proxy for the steady-state value.
[D,N]=size(z);
mPP=median(Pp,3);
logL=sum(logLnormal(z,R+C*mPP*C'));
%S=z*z'/N;
%logL=-.5*(mean(diag(P\S))+logdetP+log(2*pi));
%Naturally, this is maximized over positive semidef. P (for a given set of residuals z)
%when P=S, and then it only depends on the sample covariance of the residuals:
%maxLperSamplePerDim = -.5*(1+mean(log(eig(S)))+log(2*pi))
end

function logL=logLfast(z,Pp,C,R,A)
%Do exact for M samples, do approximate afterwards. Should be almost equal
%to the exact version, but much faster. We exploit the fact that on a
%stable system the uncertainty reaches a steady-state, so the computation
%reduces to that of the approximate version

[D,N]=size(z);
M2=20; %Default for fast filtering: 20 samples
M1=ceil(3*max(-1./log(abs(eig(A))))); %This many strides ensures ~convergence of gains before we assume steady-state
M=max(M1,M2);
M=min(M,N); %Prevent more than N, if this happens, we are not doing fast filtering

logL1=logLexact(z(:,1:M),Pp(:,:,1:M),C,R);
logL2=logLapprox(z(:,M+1:N),Pp(:,:,M+1:N),C,R);
logL=logL1+logL2;

end

function logL=logLopt(z)
%Max possible log-likelihood over all matrices P
%See refineQR for computing optimal values of Q,R that preserve z and
%maximize logL
[D2,N2]=size(z);
u=z/sqrt(N2);
S=u*u';
logdetS=log(eig(S));
logL = sum(-.5*(1+log(2*pi)+logdetS));
%Special case: if S is isotropic (all eigenvalues are the same), then log(det(S))=D2*log(trace(S)/D2) and N2*trace(S)=norm(z,'fro')^2
%Thus: logL = -.5*N2*D2*(1+log(norm(z,'fro')^2/N2)-log(D2)+log(2*pi))
%logL=N2*maxLperSample;
end
