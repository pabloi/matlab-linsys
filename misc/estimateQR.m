function [Q,R]=estimateQR(Y,C,A)
pC=pinv(C);
Z=Y(:,2:end)-C*A*pC*Y(:,1:end-1);
S1=Z(:,2:end)'*Z(:,1:end-1); %This should be -C*A*pC*R
R=-C*pinv(A)*pC*S1;
S2=Z'*Z; %This should be R+ C*A*pinv(C)*R*pinv(C)*A*C + C*Q*C'
Q=pC*(S2-R+S1*pC*A*C)*pC';

end