function logL=dataLogLikelihood(Y,U,A,B,C,D,Q,R,X0,P0)
%Evaluates the likelihood of the data under a given model

if nargin<10 || isempty(X0) || isempty(P0)
    X0=[];
    P0=[];
end
if size(X0,2)<=1 %True init state guess
[~,~,Xp,Pp,~]=statKalmanFilter(Y,A,C,Q,R,X0,P0,B,D,U,0);
else %whole filtered priors are provided, not just t=0
    Xp=X0;
    Pp=P0;
end

%'Incomplete' logLikelihood: p({y}|params) [Albert and Shadmehr 2017, eq. A1.25]
predY=C*Xp(:,1:end-1)+D*U;
z=Y-predY;
[D2,N2]=size(z);

%Fast, approximate way:
%ss=sqrtm(mean(Pp,3)); %To compute in a way that ensures the matrices are PSD, cant be done on gpu
CPpCt=C*mean(Pp,3)*C';
CPpCt=(CPpCt+CPpCt')/2; %Cheap way to ensure PSD
P=R+CPpCt; %This is a lower bound on the uncertainty of the output
% P=R+C*(A*Q*A'+Q)*C'; %This is an upper bound 
eP=eig(P);
if ~all(imag(eP)==0 & eP>0) %Sanity check, the output covariance should be positive semidef., otherwise the likelihood is not well defined
    error('Covariance matrix is not PSD, cannot compute likelihood')
end
logdetP= sum(log(eP)); %Should use:https://en.wikipedia.org/wiki/Matrix_determinant_lemma to cheapen computation (can exploit knowing C'*(R\C) and det(R) ahead of time to only need computing size(Pp) determinants


S=z*z'/N2;
%logL=-.5*N2*(trace(lsqminnorm(P,S,1e-8))+logdetP+D2*log(2*pi)); %Non-gpu ready
logL=-.5*N2*(trace(P\S)+logdetP+D2*log(2*pi)); %This line is gpu-executable
%Naturally, this is maximized over positive semidef. P (for a given set of residuals z) when P=S, and then it only depends on the sample covariance of the residuals logL=-.5*N2*(D2+log(det(S))+D2*log(2*pi))
%Whenever R is fit (optimized, as in E-M), we expect this to be the case: R should be set to the value that minimizes this quantity, thus logL is only a function of the 1-step ahead residuals
%NOTE: even though changing R will also change the filtered predictions, and hence the residuals, for any change in R that changes the filtering, there exists a change in Q such that the predictions are unchanged 
%(all that matters is that for given C, Q and C'*R*C maintain their proportionality, so the Kalman gain is unchanged, thus we can consider the optimization of R with fixed residuals)
%Special case: if S is isotropic (all eigenvalues are the same), then log(det(S))=D2*log(trace(S)/D2) and N2*trace(S)=norm(z,'fro')^2
%Thus: logL = -.5*N2*D2*(1+log(norm(z,'fro')^2/N2)-log(D2)+log(2*pi))

%Exact way: (very slow)
% minus2ly=nan(size(z,2),1);
% for i=1:size(z,2)
%     P=R+C*Pp(:,:,i)*C'; 
%     logdetP= sum(log(eig(P)));%Should use:https://en.wikipedia.org/wiki/Matrix_determinant_lemma to cheapen computation (can exploit knowing C'*(R\C) and det(R) ahead of time to only need computing size(Pp) determinants
%     zz=z(:,i);
%     minus2ly(i)=zz'*(P\zz) +logdetP + D2*log(2*pi);
% end
% logL=-.5*(sum(minus2ly)); 
%I expect the logL to be a function of two things: 
%1) the output error given the most likely states (smoothed? filtered?) 
%and 2) the innovation of states on. should try to prove it.
%In the limit of Q->0, this has to trivially resolve to some function of
%the smooth output error (i.e. least squares over the deterministic system)
end
