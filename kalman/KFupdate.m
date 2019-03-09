function [newX,newP,logL,rejectedSample,iL]=KFupdate(C,R,y,x,P,rejectZ2threshold)
%Computes the update step of the Kalman filter
%C arbitrary matrix
%R observation noise covariance. %Needs to be only PSD, but PD is HIGHLY recommended
%P prior state covariance, Needs to be PSD only
%R +C*P*C' should be strictly PD. Numerical issues some time prevent this computation being done accurately
rejectedSample=false;
%cS= coder.nullcopy(R);
cS=chol(R+C*P*C');
%iL=coder.nullcopy(cS);
iL=cS\eye(size(R));
CiL=C'*iL;
%PCiL=coder.nullcopy(CiL);
PCiL=P*CiL;


%innov=coder.nullcopy(y);
innov=y-C*x;
halfLog2Pi=0.91893853320467268;
%halfLogdetSigma=0; %Declaring type
halfLogdetSigma= sum(log(diag(cS)));
%iLy=coder.nullcopy(innov);
iLy=iL'*innov;
%z2 = 0; %Declaring type
z2=iLy'*iLy;%sum(iLy.^2,1); %z^2 scores
logL=-.5*z2 -halfLogdetSigma-size(y,1)*halfLog2Pi;

if rejectZ2threshold>0 && z2>rejectZ2threshold %Reject sample, no update
    rejectedSample=true;
    newX=x;
    newP=P;
else
    newX=x+PCiL*iLy; %P*C'*inv(S) = K
    newP = P - PCiL*PCiL'; %=(I-PCiL*CiL')*P*(I-PCil*CiL')' + K*R*K'; %For ensuring PSD-ness, would require chol(R) [precomputable] and chol(P)
end
end
