function [A,B,C,D,X,Q,R,S]=subspaceID(Y,U,d,i)
%Sub-space method indentification
%Following Shadmehr & Mussa-Ivaldi 2012
%This appears to follow Algorithm 2, in Chapter 4 of the book by
%Van Overschee, De Moor 1996 (Subspace Id for Lineary Systems),
%which the book acknowledges is a biased estimate

N=size(Y,2);
Ny=size(Y,1); %Observation dimension
Nu=size(U,1);
%i=2*round(sqrt(N*d/100)/2);%30; %For good estimation, d < i << N
if nargin<4 || isempty(i)
    i=10; %Arbitrary choice, only criterion is i > d.
    %Precision increases with large i, and large j, but when sample size is
    %finite, there is a trade-off between one and the other
end
j=N-2*i;

Y_1i=myhankel(Y,i,j);
U_1i=myhankel(U,i,j);
W_1i=[U_1i; Y_1i];
U_ip12i=myhankel(U(:,(i+1):end),i,j);
Y_ip12i=myhankel(Y(:,(i+1):end),i,j);

%O_ip1=(projectPerp(Y_ip12i,U_ip12i)/projectPerp(W_1i,U_ip12i))*W_1i;
[O_ip1,pinvU]=projectObliq(Y_ip12i,U_ip12i,W_1i);

%[T,S,V] = svd(O_ip1,'econ'); %Corresponds to W1=eye, W2=eye, which is what
%is given in Shadmehr and Mussa-Ivaldi 2012
%[T,S,V] = svd(projectPerp(O_ip1,U_ip12i),'econ'); %Corresponds to W1=eye, W2=projector, 
%which is recommended in the robust algorithm of Van Overschee and De Moor 1996
[T,S,V] = svd(O_ip1-(O_ip1*pinvU)*U_ip12i,'econ'); %SAme as above, avoids recomputing pinv(U) 

if nargin<3 || isempty(d) %Automatic figuring out of number of states...
    warning('Automatic state number detection not implemented, using 2')
    Nx=2; %Doxy
else
  Nx=d;
end

S1=sqrt(S(1:Nx,1:Nx));
T1=T(:,1:Nx);
%L_i=T1*S1; %Ljung 1999 suggests to take this matrix and estimate A,C from
%it directly (C is the first Ny rows, A can be obtained by multiplying the
%next Ny rows by the pseudo inverse of C, but that is very noisy)
%L_im1=L_i(1:(end-Ny),:);

    %This approach is ok of W1=eye:
  V_=V(1:end-1,1:Nx)';
  V__=V(2:end,1:Nx)';
  SS=sqrt(S(1:Nx,1:Nx));
  X_ip2=SS*V__;
  X_ip1=SS*V_;
  Y_ip1=Y(:,(i+1):(j+i-1));
  U_ip1=U(:,(i+1):(j+i-1));
  AB_CD=[X_ip2; Y_ip1]/[X_ip1; U_ip1];
  
  %For generic W1:
  %X_ip1=pinv(L_i)*O_ip1;
  %X_ip2=X_ip1(:,2:end);
  %AB_CD=[X_ip2; Y_ip12i(1:Ny,1:end-1)]/[X_ip1(:,1:end-1);U_ip12i(1:Nu,1:end-1)];
  
  A=AB_CD(1:Nx,1:Nx);
  B=AB_CD(1:Nx,Nx+1:end);
  C=AB_CD(Nx+1:end,1:Nx);
  D=AB_CD(Nx+1:end,Nx+1:end);
  z=X_ip2-A*X_ip1-B*U_ip1;
  w=Y_ip1-C*X_ip1-D*U_ip1;
  Q=z*z'/size(z,2);
  R=w*w'/size(w,2);
  S=z*w'/size(w,2);
  %Estimate states for all datapoints:
  X=C\(Y-D*U);
end
