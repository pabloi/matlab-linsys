function [x,P,K,z,zprctile]=KFupdate(C,R,y,x,P)


cP=mycholcov(P);
CcP=C*cP';
cS=chol(R+CcP*CcP'); %This should always be PD, so chol() should work,
%however very small (but positive) eigenvalues in R and P can lead to a
%numerically non-PD matrix, and this fails
icS=eye(size(R))/cS;
%[icS]=pinvchol(R+CcP*CcP'); %Equivalent to two lines above, but
%slightly slower, because of overhead checks of invertibility
%CicS=C'*icS;
PCicS=P*C'*icS;
K=PCicS*icS'; %K=P*C'/S;
x=x+K*(y-C*x);
I_KC=eye(size(P))-K*C;
%P=P-PCicS*PCicS';%=P-P*C'/S*C*P;%=P-K*C*P; %This expression may lead to
%non-psd covariance, since it is the subtraction of two psd matrices
I_KCcP=I_KC*cP';
P=I_KCcP*I_KCcP'+K*R*K';


%If we wanted to check sanity of the update, by evaluating if the
%innovation (of the state) is within reason given the prior expectations:
if nargout>4
    [zprctile,z]=z2prctile(y,[],C*x,icS');
elseif nargout>3
    
    z=z2score(y,[],C*x,icS');
end
end