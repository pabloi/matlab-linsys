function logLperSamplePerDim=dataLogLikelihood(Y,U,A,B,C,D,Q,R,X0,P0,method)
%Evaluates the likelihood of the data under a given model

if nargin<11 || isempty(method)
    method='approx'; %Computing approx version by default, exact is too slow
end
if nargin<10 || isempty(X0) || isempty(P0)
    X0=[];
    P0=[];
end
if size(X0,2)<=1 %True init state guess
    [~,~,Xp,Pp,~]=statKalmanFilter(Y,A,C,Q,R,X0,P0,B,D,U,0);
    %[~,~,Xp,Pp,~]=statKalmanFilterFast(Y,A,C,Q,R,X0,P0,B,D,U,[],0);
else %whole filtered priors are provided, not just t=0
    Xp=X0;
    Pp=P0;
end

%'Incomplete' logLikelihood: p({y}|params) [Albert and Shadmehr 2017, eq. A1.25]
predY=C*Xp(:,1:end-1)+D*U;
z=Y-predY;

switch method
    case 'approx'
        logLperSamplePerDim=logLapprox(z,Pp,C,R); 
    case 'exact'
        logLperSamplePerDim=logLexact(z,Pp,C,R);
    case 'max'
        logLperSamplePerDim=logLopt(z);
end

end

function logLperSamplePerDim=logLexact(z,Pp,C,R)
[D2,N2]=size(z);
%Exact way: (very slow)
minus2ly=nan(size(z,2),1);
for i=1:size(z,2)
    P=R+C*Pp(:,:,i)*C'; 
    logdetP= sum(log(eig(P)));%Should use:https://en.wikipedia.org/wiki/Matrix_determinant_lemma to cheapen computation (can exploit knowing C'*(R\C) and det(R) ahead of time to only need computing size(Pp) determinants
    eP=eig(P);
    if ~all(imag(eP)==0 & eP>0) %Sanity check, the output covariance should be positive semidef., otherwise the likelihood is not well defined
        error('Covariance matrix is not PSD, cannot compute likelihood')
    end
    zz=z(:,i);
    minus2ly(i)=zz'*(P\zz) +logdetP + D2*log(2*pi);
end
logL=-.5*(sum(minus2ly)); 
logLperSamplePerDim=logL/(N2*D2);
end

function logLperSamplePerDim=logLapprox(z,Pp,C,R)
%Faster, approximate way:
CPpCt=C*mean(Pp,3)*C';
CPpCt=(CPpCt+CPpCt')/2; %Cheap way to ensure PSD
P=R+CPpCt; 
eP=eig(P);
if ~all(imag(eP)==0 & eP>0) %Sanity check, the output covariance should be positive semidef., otherwise the likelihood is not well defined
    error('Covariance matrix is not PSD, cannot compute likelihood')
end
logdetP= mean(log(eP)); %Should use:https://en.wikipedia.org/wiki/Matrix_determinant_lemma to cheapen computation (can exploit knowing C'*(R\C) and det(R) ahead of time to only need computing size(Pp) determinants
S=z*z'/N2;
%logL=-.5*N2*(trace(lsqminnorm(P,S,1e-8))+logdetP+D2*log(2*pi)); %Non-gpu ready
logLperSamplePerDim=-.5*(mean(diag(P\S))+logdetP+log(2*pi));
%Naturally, this is maximized over positive semidef. P (for a given set of residuals z) 
%when P=S, and then it only depends on the sample covariance of the residuals:
%maxLperSamplePerDim = -.5*(1+mean(log(eig(S)))+log(2*pi))
end

function logLperSamplePerDim=logLopt(z)
%Max possible log-likelihood over all matrices P
%See refineQR for computing optimal values of Q,R that preserve z and
%maximize logL
u=z/sqrt(N2);
S=u*u';
logdetS=mean(log(eig(S)));
logLperSamplePerDim = -.5*(1+log(2*pi)+logdetS);
%Special case: if S is isotropic (all eigenvalues are the same), then log(det(S))=D2*log(trace(S)/D2) and N2*trace(S)=norm(z,'fro')^2
%Thus: logL = -.5*N2*D2*(1+log(norm(z,'fro')^2/N2)-log(D2)+log(2*pi))
%logL=N2*maxLperSample; 
end