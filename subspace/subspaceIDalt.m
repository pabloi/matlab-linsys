function [A,B,C,D,X,Q,R]=subspaceIDalt(Y,U,d,i)
%Sub-space method indentification
%Implementing Algorithm 1, Chapter 4, of the Subspace Id for Linear Systems
%(1996) book. This is an UNBIASED estimate, unlike the method implemented
%in subspaceID.m
%It incorporates *SOME* of the suggested changes from the Robust Algorithm
%in the same chapter, which leads to better results according to the
%authors
%The choice of matrices W1 and W2 are: W1= eye, W2= projector orthogonal to
%Uf. This is the matrix choice suggested in Algorithm 3 of the same
%chapter. %In general, W1 must be full rank, but W2 need only satisfy that
%rank(W_p)=rank(W_p*W2) (see code for def. of W_p).

N=size(Y,2); %Sample size
Ny=size(Y,1); %Observation dimension
%i=2*round(sqrt(N*d/100)/2);%30; %For good estimation, d < i << N
if nargin<4 || isempty(i)
    i=10; %Arbitrary choice, only criterion is i > d.
    %Precision increases with large large j, but unlike subspaceID, it is
    %unclear if larger i changes anything
end
j=N-2*i;

Y_p=myhankel(Y,i,j);
U_p=myhankel(U,i,j);
Nu=size(U,1);
W_p=[U_p; Y_p];
U_f=myhankel(U(:,(i+1):end),i,j);
Y_f=myhankel(Y(:,(i+1):end),i,j);

[O_i,pinvUf]=projectObliq(Y_f,U_f,W_p);
Z_i=projectMat(Y_f,[W_p; U_f]);

Y_pp=myhankel(Y,i+1,j);
U_pp=myhankel(U,i+1,j); %Should this be j-1?
W_pp=[U_pp;Y_pp];
Z_ip1=projectMat(Y_f(Ny+1:end,:),[W_pp; U_f(Nu+1:end,:)]);
%Z_ip1=Z_i(Ny+1:end,:); %Equivalent to the lines above(?), more efficient

%[T,S,~] = svd(O_i,'econ'); %Corresponds to W1=eye, W2=eye
%[T,S,~] = svd(projectPerp(O_i,U_f),'econ'); %Corresponds to W1=eye, W2=projector
[T,S,~] = svd(O_i-(O_i*pinvUf)*U_f,'econ'); %Same as above, avoids recomputing pinv(U) 

if nargin<3 || isempty(d) %Automatic figuring out of number of states...
    warning('Automatic state number detection not implemented, using 2')
    Nx=2; %Doxy
else
    Nx=d;
end
S1=sqrt(S(1:Nx,1:Nx));
T1=T(:,1:Nx);
L_i=T1*S1;
L_im1=L_i(1:(end-Ny),:);

pLim1=pinv(L_im1);
M=[pLim1*Z_ip1; Y_f(1:Ny,:)];
pLi=pinv(L_i);
R=[pLi*Z_i; U_f];
ACK=M*pinv(R);
A=ACK(1:Nx,1:Nx);
C=ACK(Nx+1:end,1:Nx);
K=ACK(:,Nx+1:end);

%Optional improvement: recompute L_i, for improved performance:
L_i=observabilityMatrix(A,C,i);
L_im1=L_i(1:(end-Ny),:);
pLim1=pinv(L_im1);
pLi=pinv(L_i);
S=M-[A;C]*pLi*Z_i;
K=S*pinvUf;

%Estimation of B,D from K: as done in Algorithm 1
% K1=K(1:Nx,:);
% K2=K(Nx+1:end,:);
% q=size(K1,2)/i;
% KK=nan(i*(Nx+Ny),q);
% for kk=1:i
%     KK([1:Nx]+(kk-1)*Nx,:)=K1(:,[1:q]+(kk-1)*q);
%     KK([1:Ny]+(kk-1)*Ny+i*Nx,:)=K2(:,[1:q]+(kk-1)*q);
% end
% L=[A;C]*pLi;
% M_L=[zeros(Nx,Ny) pLim1]-L(1:Nx,:);
% I_L=[eye(Ny) zeros(Ny,Ny*(i-1))]-L(Nx+1:end,:);
% N=zeros((Nx+Ny)*i,Ny*i);
% for kk=1:i %Build N:
%     N([1:Nx]+(kk-1)*Nx,1:Ny*(i-kk+1))=M_L(:,(kk-1)*Ny+1:end);
%     N([1:Ny]+(kk-1)*Ny+Nx*i,1:Ny*(i-kk+1))=I_L(:,(kk-1)*Ny+1:end);
% end
% N=N*[eye(Ny) zeros(Ny,Nx); zeros(size(L_im1,1),Ny) L_im1];
% DB=N\KK;
% D=DB(1:Ny,:);
% B=DB(Ny+1:end,:);

%Alt B,D: recommended improvement in robust algorithm
%More direct, does not estimate K to then estimate B,D.
%This is more accurate, but much slower than the method above.
%Need to debug, as results are very bad with either method.
QN=0;
L=[A;C]*pLi;
M_L=[zeros(Nx,Ny) pLim1]-L(1:Nx,:);
I_L=[eye(Ny) zeros(Ny,Ny*(i-1))]-L(Nx+1:end,:);
F=[eye(Ny) zeros(Ny,Nx); zeros(size(L_im1,1),Ny) L_im1];
for kk=1:i
   N1=M_L(:,(kk-1)*Ny+1:end);
   N2=I_L(:,(kk-1)*Ny+1:end);
   aux=kron(U_f([1:Nu]+(kk-1)*Nu,:)',[N1;N2]*F);
   QN=QN+aux;
end
DB=pinv(QN)*S(:);
D=reshape(DB(1:Ny*Nu),Ny,Nu);
B=reshape(DB(Ny*Nu+1:end),Nx,Nu);
 
%Estimate noise levels:
zw=M-ACK*R; %=S-K*U_f;
z=zw(1:Nx,:);
w=zw(Nx+1:end,:);

Q=z*z'/size(z,2);
R=w*w'/size(w,2);
S=z*w'/size(w,2);
%Estimate states for all datapoints:
X=C\(Y-D*U);

end
