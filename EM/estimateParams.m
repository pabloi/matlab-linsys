function [A,B,C,D,Q,R,x0,P0]=estimateParams(Y,U,X,P,Pt)
%M-step of EM estimation for LTI-SSM
%INPUT:
%Y = output of the system, D2 x N
%U = input of the system, D3 x N
%X = state estimates of the system (Kalman-smoothed), D1 x N
%P = covariance of states (Kalman-smoothed), D1 x D1 x N
%Pp = covariance of state transitions (Kalman-smoothed), D1 x D1 x (N-1),
%evaluated at k+1|k
%See Cheng and Sabes 2006, Ghahramani and Hinton 1996

%Doxy:
[A,B,Q] = estimateAB(X, U);
[C,D,R] = estimateCD(Y,X, U);
x0=X(:,1);
P0=[];
return

%As it should be: (doesnt work)
N=size(X,1);
%A,B:
xu=X(:,1:end-1)*U(:,1:end-1)';
uu=U(:,1:end-1)*U(:,1:end-1)';
xu1=X(:,2:end)*U(:,1:end-1)';
%AB=[sum(Pt,3) xu1]/[sum(P(:,:,1:end-1),3) xu; xu' uu];
O=[sum(P(:,:,1:end-1),3) xu; xu' uu];
AB=[sum(Pt,3) xu1]*pinv(O+1e-9*eye(N+1));
A=AB(:,1:size(X,1));
B=AB(:,size(X,1)+1:end);

%C,D:
xu=X*U';
uu=U*U';
%CD=[Y*X' Y*U']/[sum(P(:,:,1:end-1),3) xu; xu' uu];
O=[sum(P,3) xu; xu' uu];
CD=[Y*X' Y*U']*pinv(O+1e-9*eye(N+1));
C=CD(:,1:size(X,1));
D=CD(:,size(X,1)+1:end);

%Q,R:
Q=mean(P(:,:,2:end),3) - A*mean(Pt,3)' -B*xu1'; %A*P' is an ill-defined product here I mean the matrix product for each 2D slice along the third dim of Pp 
Q=.5*(Q+Q');
Q=Q+1e-5*eye(size(Q));
R=mean(Y*Y'-C*X*Y'-D*U*Y',2); %This estimate is weird: does not result in a symmetric matrix
R=.5*(R+R');
R=R+1e-5*eye(size(R));

%x0,P0:
x0=X(:,1);
P0=P(:,:,1);%-x0*x0'; %Ghahramani 1996 subtract the x0 term, Cheng 2006 doesnt
