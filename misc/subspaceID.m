function [A,B,C,D,X,Q,R]=subspaceID(Y,U,d)
%Sub-space method indentification
%Following Shadmehr & Mussa-Ivaldi 2012

N=size(Y,2);
i=2;
j=N-2*i;

Y_1i=myhankel(Y,i,j);
U_1i=myhankel(U,i,j);
W_1i=[U_1i; Y_1i];
U_ip12i=myhankel(U(:,(i+1):end),i,j);
Y_ip12i=myhankel(Y(:,(i+1):end),i,j);

O_ip1=(projectPerp(Y_ip12i,U_ip12i)/projectPerp(W_1i,U_ip12i))*W_1i;
[~,S,V] = svd(O_ip1,'econ');
if nargin<3 %Automatic figuring out of number of states...
%figure
%sd=(diag(S));
%plot(cumsum(sd)/sum(sd))
%set(gca,'YScale','log')
Nx=2;
else
  Nx=d;
end
X=sqrt(S(1:Nx,1:Nx))*V(:,1:Nx)';
V_=V(1:end-1,1:Nx)';
V__=V(2:end,1:Nx)';
X_ip2=sqrt(S(1:Nx,1:Nx))*V__;
X_ip1=sqrt(S(1:Nx,1:Nx))*V_;
Y_ip1=Y(:,(i+1):(j+i-1));
U_ip1=U(:,(i+1):(j+i-1));
AB_CD=[X_ip2; Y_ip1]/[X_ip1; U_ip1];
A=AB_CD(1:Nx,1:Nx);
B=AB_CD(1:Nx,Nx+1:end);
C=AB_CD(Nx+1:end,1:Nx);
D=AB_CD(Nx+1:end,Nx+1:end);
z=X_ip2-A*X_ip1-B*U_ip1;
w=Y_ip1-C*X_ip1-D*U_ip1;
Q=z*z'/N;
R=w*w'/N;
%Estimate states for all datapoints:
X=C\(Y-D*U);
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
