function [newX,newcPt,logL,rejectedSample,icS]=KFupdateSQRT(Ct,cRt,y,x,cPt,rejectZ2threshold)
%Computes the update step of the Kalman filter
%C arbitrary matrix
%R observation noise covariance. %Needs to be only PSD, but PD is HIGHLY recommended
%P prior state covariance, Needs to be PSD only
%R +C*P*C' should be strictly PD. Numerical issues some time prevent this computation being done accurately

rejectedSample=false;
[~,cSt]=qr([cRt; cPt*Ct],0);
icSt=eye(size(cRt))/cSt';
icS=icSt';
CiL=Ct*icS;
PCiL=(cPt'*cPt)*CiL;

innov=y-Ct'*x;
iLy=icSt*innov;
z2=iLy'*iLy; %sum(iLy.^2,1); %z^2 scores
if nargout>4 %Expensive to evaluate, doing only if needed
    halfLog2Pi=0.91893853320467268;
    halfLogdetSigma= sum(log(diag(abs(cSt))));
    logL=-.5*z2 -halfLogdetSigma-size(y,1)*halfLog2Pi;
end

if rejectZ2threshold>0 && z2>rejectZ2threshold %Reject sample, no update
    rejectedSample=true;
    newX=x;
    newcPt=cPt;
else
    newX=x+PCiL*iLy; %P*C'*inv(S) = K
    %newP = P - PCiL*PCiL'; %This is fine if no ill-conditioned matrices
    %are ever present, such that PCiL*PCiL' has a larger
    %diagonal than P because of numerical issues.
    K=(PCiL*icSt);
    %Kt=iL*PCiL';
    I=eye(size(cPt));
    H=(I-K*Ct');
    %Ht=(I-Ct*Kt);
    %HcP=H*cP;
    %KcR=K*cR;
    %newP = HcP*HcP' + KcR*KcR'; %For ensuring PSD-ness, would require chol(R) [precomputable] and chol(P)
    [~,newcPt]=qr([cPt*H';cRt*K'],0);
end
end
