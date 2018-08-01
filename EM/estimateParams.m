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

%x0,P0:
x0=X(:,1);
P0=P(:,:,1);

%A,B:
xu=X(:,1:end-1)*U(:,1:end-1)';
uu=U(:,1:end-1)*U(:,1:end-1)';
xu1=X(:,2:end)*U(:,1:end-1)';
xx=X(:,1:end-1)*X(:,1:end-1)';
xx1=X(:,2:end)*X(:,1:end-1)';
O=[sum(P(:,:,1:end-1),3)+xx xu; xu' uu];
%AB=[sum(Pt,3)+xx1 xu1]*pinv(O,1e-8);
AB=[sum(Pt,3)+xx1 xu1]/O;
%AB=lsqminnorm(O,[sum(Pt,3)+xx1 xu1]',tol)'; %More efficient than commented line above
%Notice that in absence of uncertainty in states, this reduces to
%[A,B]=X+/[X;U], where X+ is X one step in the future
A=AB(:,1:size(X,1));
B=AB(:,size(X,1)+1:end);

%C,D:
xu=X*U';
uu=U*U';
xx=X*X';
O=[sum(P,3)+xx xu; xu' uu];
%CD=[Y*X' Y*U']*pinv(O,tol);
%CD=lsqminnorm(O,[X;U]*Y',tol)'; %More efficient than line above
CD=Y*[X; U]'/O;
%Notice that in absence of uncertainty in states, this reduces to [C,D]=Y/[X;U]
C=CD(:,1:size(X,1));
D=CD(:,size(X,1)+1:end);

%Q,R: 
%Adaptation of Shumway and Stoffer 1982: (there B=D=0 and C is fixed), but
%consistent with Ghahramani and Hinton 1996, and Cheng and Sabes 2006
z=Y-C*X-D*U;
w=X(:,2:end)-A*X(:,1:end-1)-B*U(:,1:end-1);
Q=(w*w')/size(w,2)+mean(P,3)-A*mean(Pt,3)';
Q=positivize(Q); %Expression above should be symmetric and PSD, but may not be because of numerical issues
R=z*z'/size(z,2) +C*mean(P,3)*C';
R=positivize(R); %Expression above should be symmetric and PSD, but may not be because of numerical issues

iP=pinv(P0,1e-8);
iP0=iP+C'*(R\C);
P0=Q+A*(iP0\A'); %P0=Q+A*P0*A can be a proxy