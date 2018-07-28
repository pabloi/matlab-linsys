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

%Doxy: this is equivalent to the true M-step if all covariances are null
% [A,B,Q] = estimateAB(X, U);
% [C,D,R] = estimateCD(Y,X, U);
% x0=X(:,1);
% P0=[];
% return


%As it should be:
tol=1e-8;
%x0,P0:
x0=X(:,1);
P0=P(:,:,1);%-x0*x0'; %Ghahramani 1996 subtract the x0 term, Cheng 2006 doesnt

%A,B:
xu=X(:,1:end-1)*U(:,1:end-1)';
uu=U(:,1:end-1)*U(:,1:end-1)';
xu1=X(:,2:end)*U(:,1:end-1)';
xx=X(:,1:end-1)*X(:,1:end-1)';
xx1=X(:,2:end)*X(:,1:end-1)';
O=[sum(P(:,:,1:end-1),3)+xx xu; xu' uu];
%AB=[sum(Pt,3)+xx1 xu1]*pinv(O,1e-8);
AB=lsqminnorm(O,[sum(Pt,3)+xx1 xu1]',tol)'; %More efficient than commented line above
A=AB(:,1:size(X,1));
B=AB(:,size(X,1)+1:end);

%C,D:
xu=X*U';
uu=U*U';
xx=X*X';
O=[sum(P,3)+xx xu; xu' uu];
%CD=[Y*X' Y*U']*pinv(O,tol);
CD=lsqminnorm(O,[X;U]*Y',tol)'; %More efficient than line above
C=CD(:,1:size(X,1));
D=CD(:,size(X,1)+1:end);

%Q,R: (as featured on Cheng and Sabes 2006)
%Q=(sum(P(:,:,2:end),3)+xx-x0*x0' - A*(sum(Pt,3)'+xx1) -B*xu1')/size(Pt,3); 
%Q=.5*(Q+Q'); %Needed because Q is not symmetric as-is, still, there is no
%guarantee of it being PSD
z=Y-C*X+D*U;
%R=(z*Y')/size(Y,2); %This estimate is weird: does not result in a symmetric matrix
%R=.5*(R+R'); %Needed because Q is not symmetric as-is, still, there is no
%guarantee of it being PSD

%I think Q,R should be, in absence of state uncertainty: (this is consistent with Cheng & Sabes own code:
%https://sabeslab.cin.ucsf.edu/wiki/Public:Notes)
R=z*z'/size(z,2)+1e-7*eye(size(z,1));
w=X(:,2:end)-A*X(:,1:end-1)-B*U(:,1:end-1);
Q=(w*w')/size(w,2)+1e-7*eye(size(w,1));
%Although this has issues of convergence: if Q starts too small, the
%EM algorithm is stuck (because of small Q, the smoothed estimate of X is
%almost deterministic, which in turn makes all the residuals very small,
%which leads to a small estimate of Q).

%If there is state uncertainty:
%Q=(w*w')/size(w,2) - mean(P(:,:,2:end),3) - A*mean(Pt,3)'-mean(Pt,3)*A' - A*mean(P(:,:,1:end-1),3)*A' +1e-3*eye(size(w,1));
%R=z*z'/size(z,2) -C*mean(P,3)*C' +1e-4*eye(size(z,1));