function [A,B,C,D,Q,R,x0,P0]=estimateParams(Y,U,X,P,Pt)
%M-step of EM estimation for LTI-SSM
%INPUT:
%Y = output of the system, D2 x N
%U = input of the system, D3 x N
%X = state estimates of the system (Kalman-smoothed), D1 x N
%P = covariance of states (Kalman-smoothed), D1 x D1 x N
%Pp = covariance of state transitions (Kalman-smoothed), D1 x D1 x (N-1),
%evaluated at k+1|k
%See Cheng and Sabes 2006, Ghahramani and Hinton 1996, Shumway and Stoffer 1982

%This is equivalent to the true M-step if no uncertainty in X (P=Pt=0)
% [A,B,Q] = estimateAB(X, U);
% [C,D,R] = estimateCD(Y,X, U);
% x0=X(:,1);
% P0=[];
% return

%True M-step
%tol=1e-8;

%Define vars:
SP1=sum(P(:,:,1:end-1),3);
SP2=sum(P(:,:,2:end),3);
SP=sum(P,3);
SPt=sum(Pt,3);
[D1,N]=size(X);


%x0,P0:
x0=X(:,1);
P0=P(:,:,1);

%A,B:
xu=X(:,1:end-1)*U(:,1:end-1)';
uu=U(:,1:end-1)*U(:,1:end-1)';
xu1=X(:,2:end)*U(:,1:end-1)';
xx=X(:,1:end-1)*X(:,1:end-1)';
xx1=X(:,2:end)*X(:,1:end-1)';
O=[SP1+xx xu; xu' uu];
%AB=[sum(Pt,3)+xx1 xu1]*pinv(O,1e-8);
AB=[SPt+xx1 xu1]/O; %More efficient than above
%AB=lsqminnorm(O,[SPt+xx1 xu1]',tol)'; %More stable than above
%Notice that in absence of uncertainty in states, this reduces to
%[A,B]=X+/[X;U], where X+ is X one step in the future
A=AB(:,1:D1);
B=AB(:,D1+1:end);

%C,D:
xu=X*U';
uu=U*U';
xx=X*X';
O=[SP+xx xu; xu' uu];
%CD=[Y*X' Y*U']*pinv(O,tol);
%CD=lsqminnorm(O,[X;U]*Y',tol)'; %More efficient than line above
CD=Y*[X; U]'/O;
%Notice that in absence of uncertainty in states, this reduces to [C,D]=Y/[X;U]
C=CD(:,1:D1);
D=CD(:,D1+1:end);

%Q,R: 
%Adaptation of Shumway and Stoffer 1982: (there B=D=0 and C is fixed), but
%consistent with Ghahramani and Hinton 1996, and Cheng and Sabes 2006
z=Y-C*X-D*U;
w=X(:,2:N)-A*X(:,1:N-1)-B*U(:,1:N-1);

%Cheap outlier rejection:
n=sqrt(sum(w.^2,1));
n99=prctile(n,99);
%w(:,n>n99)=repmat(mean(w,2),1,sum(n>n99));
%Q=(w*w'+SP2-2*A*SPt'+A*SP1*A')/(N-1);
%Alt: Robust covariance estimation:
Q = robustcov(w') +(SP2-2*A*SPt'+A*SP1*A')/(N-1);
%
Q=positivize(Q); %Expression above should be symmetric and PSD, but may not be because of numerical issues
R=(z*z'+C*SP*C')/N;
%R=positivize(R); %Expression above should be symmetric and PSD, but may not be because of numerical issues
%Diagonal imposition:
%R=trace(R)*eye(size(R))/size(R,1); %Isotropic noise
%r=diag(R);
%r(r<0)=0;
%R=diag(r);
%Q=eye(D1)*.0005;

iP=pinv(P0,1e-8);
iP0=iP+C'*(R\C);
P0=Q+A*(iP0\A'); %P0=Q+A*P0*A can be a proxy
P0=positivize(P0);

