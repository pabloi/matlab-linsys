function [Y,U,A,B,C,D,Q,R,x0,Yoff,Ysquared,Yexp,Ylap]=generateSyntheticData(dummyArg)
ny=180;
nx=2;
A=diag(exp(-1./[30;150]));
B=1-diag(A); %All states tend to 1 under input response, WLOG.
C=(2*rand(ny,nx)-1);
C=(C.^2).*sign(C);
for i=1:nx
  C(:,i)=reshape(conv2(reshape(C(:,i),12,15),ones(3,3)/9,'same'),ny,1); %To get some structure in columns of C
end
D=(2*rand(ny,1)-1);
D=(D.^2).*sign(D);
D(:)=conv2(reshape(D,12,15),ones(3,3)/9,'same');
U=[zeros(1,150) ones(1,900) zeros(1,600)];
Q=5*diag(B.^2); %So that noise scales gracefully with the terms in B
R=.005*randn(ny);
R=R*R';
x0=zeros(nx,1);
[~,X]=fwdSim(U,A,B,C,D,x0,Q,R);
Yoff=.1*rand(ny,1)-.05; %Some small values to simulate bad bias removal
Yoff(:)=conv2(reshape(Yoff,12,15),ones(3,3)/9,'same');
Ym=C*X(:,1:end-1)+D*U+Yoff;
cR=mycholcov(R);

%Add different types of noise, with 0 mean and covariance R
Ysquared  = Ym + sign(Ym).*(cR'*(randn(size(C,1),size(X,2)-1).^2 -1)/sqrt(2)); %Signed, de-meaned (shifted), squared-gaussian noise
Yexp  = Ym + sign(Ym).*(cR'*(exprnd(1,size(C,1),size(X,2)-1)-1)); %Signed, de-meaned, exponential noise
Y = Ym + cR'*randn(size(C,1),size(X,2)-1); %Gaussian noise ~N(0,R)
Ylap  = Ym + cR'*(sign(randn(size(C,1),size(X,2)-1)).*(exprnd(1,size(C,1),size(X,2)-1))/sqrt(2)); %Laplacian noise

%See that data has a 'mean' of Ym, and only noise changed
%figure; plot(Y(1:10,:)'); hold on; set(gca,'ColorOrderIndex',1); plot(Ym(1:10,:)');
%set(gcf,'Name',['Mean error: ' num2str(sum(abs(mean(Y-Ym,2)))) ', variance of data: ' num2str(sum(var(Y,[],2)))])
end
