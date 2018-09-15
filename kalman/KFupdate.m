function [x,P,z]=KFupdate(C,R,y,x,P)

S=R+C*P*C';
cS=chol(S);
icS=cS\eye(size(S));
%K=P*C'/S;%=pinv(pinv(P)+CtRinvC);
CicS=C'*icS;
PCicS=P*CicS;
P=P-PCicS*PCicS';%=P-P*C'/S*C*P;%=P-K*C*P;
x=x+PCicS*icS'*(y-C*x);%=x+K*(y-C*x);

if nargout>2
    %If we wanted to check sanity of the update, by evaluating if the
    %innovation (of the state) is within reason given the prior expectations:
    z=z2score(y,[],C*x,icS');
end
end