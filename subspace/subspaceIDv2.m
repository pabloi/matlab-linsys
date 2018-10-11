function [A,B,C,D,X,Q,R]=subspaceIDv2(Y,U,d)
%Sub-space method indentification
%Following Shadmehr & Mussa-Ivaldi 2012

[Ny,N]=size(Y);

i=20; %Should always be even
j=N-2*i;

Y_1i=myhankel(Y,i,j);
U_1i=myhankel(U,i,j);
W_1i=[U_1i; Y_1i];
U_ip12i=myhankel(U(:,(i+1):end),i,j);
Y_ip12i=myhankel(Y(:,(i+1):end),i,j);

%Output that can be explained by a lagged-history of the output and the input
O_ip1=(projectPerp(Y_ip12i,U_ip12i)/projectPerp(W_1i,U_ip12i))*W_1i;
%O_ip1=(projectPerp(Y_ip12i,U_ip12i)/W_1i)*W_1i; %Not quite the same as above
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

sz=i-1; %Needs to be larger than Nx for unique solution
P2=permute(reshape(P(Ny+1:Ny*sz,1:Nx),Ny,sz-1,Nx),[1,3,2]);
IA=C\P2(:,:); %This should result in a matrix of the form [A; A^2; ...; A^(Nx-1)];
%IA=reshape(IA,Nx,Nx,sz-1); %  A + A^2 + A^3 + .. + A^(Nx-1)
%A=unfoldMatrix(sum(IA,3),ones(1,i-1));
%A=real(A); %Eliminating numerical issues from unfolding with complex eigenvalues
A=fitMatrixPowers(IA');
B=(X_ip2-A*X_ip1)/U_ip1;

%Residuals:
z=X_ip2-A*X_ip1-B*U_ip1;
Q=z*z'/N;
end

function H=myhankel(A,i,j)
  H=nan(i*size(A,1),j);
  for l=1:j
    a=A(:,l:l+i-1);
    H(:,l)=a(:);
  end
end

function Ap=projectPerp(A,B)
  Ap=A-(A/B)*B;
end

function IT=geomMat(T,n)
for i=1:n
  IT(:,:,i)=T^(i-1);
end
end
