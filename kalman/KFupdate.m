function [x,P,K,z]=KFupdate(C,R,y,x,P)

S=R+C*P*C';
%cS=chol(S);
%icS=eye(size(S))/cS;
[icS,cS]=pinvchol(S);
%K=P*C'/S;%=pinv(pinv(P)+CtRinvC);
CicS=C'*icS;
PCicS=P*CicS;
P=P-PCicS*PCicS';%=P-P*C'/S*C*P;%=P-K*C*P;
K=PCicS*icS';
x=x+K*(y-C*x);

if nargout>3
    %If we wanted to check sanity of the update, by evaluating if the
    %innovation (of the state) is within reason given the prior expectations:
    z=z2score(y,[],C*x,icS');
end
end