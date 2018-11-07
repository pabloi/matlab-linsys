function [newX,newP,K,logL,rejectedSample,icS]=KFupdate(C,R,y,x,P,rejectZ2threshold,~)
%Computes the update step of the Kalman filter
%C arbitrary matrix
%R observation noise covariance. %Needs to be only PSD, but PD is HIGHLY recommended
%P prior state covariance, Needs to be PSD only
%R +C*P*C' should be strictly PD. Numerical issues some time prevent this computation being done accurately

rejectFlag=true;
if nargin<6 || isempty(rejectZ2threshold)
    rejectFlag=false;
    rejectedSample=false;
end

%cP=mycholcov(P); CcP=C*cP'; %Need P to be PSD only
%[icS]=pinvchol(R+CcP*CcP'); %Equivalent to two lines above, but
%slightly slower, because of overhead checks of invertibility
[icS,cS]=pinvchol(R+C*P*C'); %Computing inv(S), with S=R+C*P*C'
if size(icS,2)<size(R,1)
    warning('KFudpate:nonPDmatrix','R+C*P*C^t was not strictly definite. This can end badly.')
end
CicS=C'*icS;
%CinvS=CicS*icS';
PCicS=P*CicS;
K=PCicS*icS'; %=P*CinvS; %K=P*C'/S; %No need to compute this

innov=y-C*x;
if nargout>3 || rejectFlag %Compute log-likelihood of observation given prior:
    [logL,z2]=logLnormal(innov,[],icS');
    if rejectFlag && z2>rejectZ2threshold %Reject sample, no update
        rejectedSample=true;    K=zeros(size(C))';
        return
    end
end

%CtinvSinnov=CinvS*innov; %=C'*inv(S)
newX=x+K*innov; %P*C'*inv(S) = K
%I_KC=eye(size(P))-K*C; %P=P-PCicS*PCicS';%=P-K*C*P;
%This expression may lead to non-psd covariance, since it is the subtraction of two psd matrices
%I_KCcP=I_KC*cP';
%if nargin<7
%    KcR=mycholcov(K*R*K');
%else
%    KcR=cR*K'; %If chol decomp of R is given, this is cheaper
%end
%newP=I_KCcP*I_KCcP'+KcR'*KcR; = (I-KC)*P*(I-KC)' + K*R*K = P-K*(R+C*P*C')*K;
newP = P - PCicS*PCicS';
end
