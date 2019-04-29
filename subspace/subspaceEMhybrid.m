function [A,B,C,D,X,Q,R]=subspaceEMhybrid(Y,U,d,i)
%Hybrid of sub-space method indentification and EM
%Implementing Algorithm 1, Chapter 4, of the Subspace Id for Linear Systems
%(1996) book to estimate A only. This is *IN THEORY* an UNBIASED estimate, 
%unlike the method implemented in subspaceID.m
% A single step of EM is then used to estimate all other parameters
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

%Y_pp=myhankel(Y,i+1,j);
%U_pp=myhankel(U,i+1,j); %Should this be j-1?
%W_pp=[U_pp;Y_pp];
%Z_ip1=projectMat(Y_f(Ny+1:end,:),[W_pp; U_f(Nu+1:end,:)]);
Z_ip1=Z_i(Ny+1:end,:); %Equivalent to the lines above(?), more efficient

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
M=[pLim1*Z_ip1];
pLi=pinv(L_i);
R=[pLi*Z_i; U_f];
AK=M*pinv(R);
A=AK(1:Nx,1:Nx);

%EM-steps:
opts.fixA=A;
opts.Niter=50; %A few iters
opts.verbose=false;
[A,B,C,D,Q,R,X,P,bestLogL,outLog]=EM(Y,U,Nx,opts);

end
