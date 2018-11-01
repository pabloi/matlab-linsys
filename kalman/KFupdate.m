function [x,P,K,z,zprctile,rejectedSample]=KFupdate(C,R,y,x,P,rejectZthreshold)
%Computes the update step of the Kalman filter
%C arbitrary matrix
%R observation noise covariance. %Needs to be only PSD, but PD is HIGHLY recommended
%P prior state covariance, Needs to be PSD only
%R +C*P*C' should be strictly PD. Numerical issues some time prevent this computation being done accurately

rejectFlag=true;
if nargin<6
    rejectFlag=false;
    rejectedSample=false;
end

cP=mycholcov(P); CcP=C*cP'; %Need P to be PSD only
%cS=chol(R+CcP*CcP'); %This HAS to be PD, if it fails, there is some issue.
%Best case scenario, a numerical issue makes an indefinite matrix out of the sum of the two PSD matrices
%icS=eye(size(R))/cS;
[icS]=pinvchol(R+CcP*CcP'); %Equivalent to two lines above, but
%slightly slower, because of overhead checks of invertibility
if size(icS,2)<size(R,1)
    warning('KFudpate:nonPDmatrix','R+C*P*C^t was not strictly definite. This can end badly.')
end
PCicS=P*C'*icS;
K=PCicS*icS'; %K=P*C'/S;

%If we wanted to check sanity of the update, by evaluating if the
%innovation (of the state) is within reason given the prior expectations:
%Get z-score and evaluate if sample is an outlier:
if nargout>4 || rejectFlag
    [zprctile,z]=z2prctile(y,[],C*x,icS');
    if rejectFlag && z>rejectZthreshold
        %Sample rejected, no update
        rejectedSample=true;
        K=zeros(size(C))';
        return
    end
elseif nargout>3
    z=z2score(y,[],C*x,icS');
end

x=x+K*(y-C*x);
I_KC=eye(size(P))-K*C;
%P=P-PCicS*PCicS';%=P-P*C'/S*C*P;%=P-K*C*P; %This expression may lead to
%non-psd covariance, since it is the subtraction of two psd matrices
I_KCcP=I_KC*cP';
KcR=mycholcov(K*R*K'); %Alt: cR=mycholcov(R); KcR=K*cR'; More precise, but involves the chol() decomp of R. Ok if we did it in statKalmanFilter
P=I_KCcP*I_KCcP'+KcR'*KcR;
end
