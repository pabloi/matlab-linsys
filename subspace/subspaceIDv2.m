function [A,B,C,D,X,Q,R]=subspaceIDv2(Y,U,d)
%Sub-space method indentification
%Following Shadmehr & Mussa-Ivaldi 2012
%error('This is shit. None of the methods returns better behavior than subspaceID, despite that method being very biased. Especially bad in the presence of non-zero Q. Trying to correct for bias makes everything worse. For some reason estimateTransitionMatrix works well but this does not. Correlated noise?')
[Ny,N]=size(Y);

i=40; %Should always be even
j=N-2*i;

Y_1i=myhankel(Y,i,j);
U_1i=myhankel(U,i,j);
W_1i=[U_1i; Y_1i];
U_ip12i=myhankel(U(:,(i+1):end),i,j);
Y_ip12i=myhankel(Y(:,(i+1):end),i,j);

%Output that can be explained by a lagged-history of the output and the input
O_ip1=(projectPerp(Y_ip12i,U_ip12i)/projectPerp(W_1i,U_ip12i))*W_1i;
[P,S,V] = svd(O_ip1,'econ');
if nargin<3 %Automatic figuring out of number of states...
  Nx=2;
else
  Nx=d;
end
X=(S(1:Nx,1:Nx))*V(:,1:Nx)';
V_=V(1:end-1,1:Nx)';
V__=V(2:end,1:Nx)';
X_ip2=(S(1:Nx,1:Nx))*V__;
X_ip1=(S(1:Nx,1:Nx))*V_;
Y_ip1=Y(:,(i+1):(j+i-1));
U_ip1=U(:,(i+1):(j+i-1));

%As presented in Shadmehr and Mussa-Ivaldi:
%AB=X_ip2/[X_ip1;U_ip1];
%A=AB(1:Nx,1:Nx);
%B=AB(1:Nx,Nx+1:end);
CD=Y_ip1/[X_ip1;U_ip1];
C=CD(:,1:Nx);
D=CD(:,Nx+1:end);

%Estimate states for all datapoints:
X=C\(Y-D*U);

%Residuals:
w=Y_ip1-C*X_ip1-D*U_ip1;
R=w*w'/N;

% Opt 1: solve a single polynomial of the matrix A
sz=i-1; %Needs to be larger than Nx for unique solution
P2=permute(reshape(P([Ny+1]:[Ny*sz],1:Nx),Ny,sz-1,Nx),[1,3,2]);
IA=C\P2(:,:);
IA=reshape(IA,Nx,Nx,sz-1); %  =cat(3,A,A^2, A^3, ..., A^(sz-1))
%A=matrixPolyRoots(sum(IA,3),ones(1,sz-1));
%A=real(A); %Eliminating numerical issues from unfolding with complex eigenvalues
% Opt 2: fit each power of the matrix A that was estimated, not considering a bias term:
ord=1;
IA=reshape(permute(IA(:,:,1:ord),[1,3,2]),Nx*ord,Nx);
A=fitMatrixPowers(IA,true);
% Opt 3: take the estimated states, and estimate the transition matrix (this works in testEstimMatrix under the independent noise of state estimation assumption)
%A=estimateTransitionMatrixv2(projectPerp(X,U),ord);
B=(X_ip2-A*X_ip1)/U_ip1;

%Residuals:
z=X_ip2-A*X_ip1-B*U_ip1;
Q=z*z'/N;
end
