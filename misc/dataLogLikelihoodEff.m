function logL=dataLogLikelihoodEff(Y,U,A,B,C,D,Q,Rinv,X0,P0)
%Evaluates the likelihood of the data under a given model

if nargin<10 || isempty(X0) || isempty(P0)
    X0=[];
    P0=[];
end
if size(X0,2)<=1 %True init state guess
    [~,~,Xp,Ppinv,~]=statKalmanFilter(Y,A,C,Q,R,X0,P0,B,D,U,0);
else %whole filtered priors are provided, not just t=0
    Xp=X0;
    Ppinv=P0;
end

%'Incomplete' logLikelihood: p({y}|params) [Albert and Shadmehr 2017, eq. A1.25]
predY=C*Xp(:,1:end-1)+D*U;
z=Y-predY;
[D2,N2]=size(z);

%Fast, approximate way:
%T=R+C*mean(Pp,3)*C'; %This is a lower bound on the uncertainty of the output
mPi=mean(Ppinv,3);
RinvC=Rinv*C;
Tinv=Rinv-RinvC*(C'*RinvC+mPi)\RinvC';
% P=R+C*(A*Q*A'+Q)*C'; %This is an upper bound 
logdetT= -sum(log(eig(Tinv))); 
%minus2ly=sum(z.*(P\z)) +logdetP + D2*log(2*pi);
%logL=-.5*sum(sum(z.*lsqminnorm(P,z,1e-8)) +logdetP + D2*log(2*pi));
%sum(minus2ly) should be MINIMIZED when P=z*z'/size(z,2)
%logL=-.5*(trace(z'*inv(P)*z)+N2*logdetP+N2*D2*log(2*pi));
%logL=-.5*(trace(inv(P)*z*z')+N2*logdetP+N2*D2*log(2*pi));
S=z*z'/N2;
logL=-.5*N2*(trace(Tinv*S)+logdetT+D2*log(2*pi));%Naturally, this is maximized when S=T
%Exact way: (very slow)
% minus2ly=nan(size(z,2),1);
% for i=1:size(z,2)
%     %T=R+C*Pp(:,:,i)*C'; 
%     Tinv=Rinv-RinvC*(C'*RinvC+Ppinv(:,:,i))\RinvC'; ...
%     logdetT= -sum(log(eig(Tinv))); 
%     zz=z(:,i);
%     minus2ly(i)=zz'*(Tinv*zz) +logdetT + D2*log(2*pi);
% end
% logL=-.5*(sum(minus2ly)); 


%I expect the logL to be a function of two things: 
%1) the output error given the most likely states (smoothed? filtered?) 
%and 2) the innovation of states on. should try to prove it.
%In the limit of Q->0, this has to trivially resolve to some function of
%the smooth output error (i.e. least squares over the deterministic system)
end


%Approximate logLikelihood: p({y},\hat{x}|params) -> Notice that this
%results in different likelihoods if we transform the state space and the
%parameters accordingly (i.e. not scale invariant). See Shumway and Stoffer
%--------------------------
%State transition likelihood
%w=X(:,2:end)-A*X(:,1:end-1)-B*U(:,1:size(X,2)-1);
%[D1,N1]=size(w);
%logdetQ=sum(log(eig(Q))); %More efficient than log(det(Q)), no over/underflow issues. 
%minus2lX=sum(w.*(Q\w)) +logdetQ + D1*log(2*pi); %This can't be, as is
%scale variant: if we rescale all the states to 1/10 (and all relate
%parameters accordingly, e.g. Q is 1/100 of prev value), then two systems
%that are the same up to a scaling of states will have different
%likelihoods. But if we suppress the logdetQ term (which gives the scale
%issue) then larger Q's are always better, everything else being the same.
%Output likelihoods
%z=Y-C*X(:,1:size(Y,2))-D*U(:,1:size(Y,2));
%[D2,N2]=size(z);
%logdetR= sum(log(eig(R))); %More efficient than log(det(R)), no over/underflow issues.
%minus2lY=sum(z.*(R\z)) +logdetR + D2*log(2*pi);
%Total:
%logL=-.5*(sum(minus2lX)+sum(minus2lY));