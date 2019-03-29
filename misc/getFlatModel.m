function [J,B,C,D,Q,R,logLperSamplePerDim]=getFlatModel(Y,U,opts)
nx=0;
nu=size(U,1);
ny=size(Y,1);
  if nargin<3
    opts=struct();
    opts=processEMopts(opts,nu,nx,ny);
  end
  bad=any(isnan(Y),1);
  Y=Y(:,~bad);
  U=U(:,~bad);
J=0;
B=zeros(1,size(U,1));
Q=0;
C=ones(size(Y,1),1);
D=Y/U;
res=Y-D*U;
Raux=res*res'/size(Y,2);

%If some outputs were excluded:
R=diag(inf(size(Y,1),1));
R(opts.includeOutputIdx,opts.includeOutputIdx)=Raux(opts.includeOutputIdx,opts.includeOutputIdx);

%Get logL:
warning('off','statKF:logLnoPrior'); %Enforcing improper prior for parameter search
logLperSamplePerDim=dataLogLikelihood(Y(opts.includeOutputIdx,:),U,J,B,C(opts.includeOutputIdx,:),D(opts.includeOutputIdx,:),Q,R(opts.includeOutputIdx,opts.includeOutputIdx),[],[],'exact');
warning('on','statKF:logLnoPrior'); %Enforcing improper prior for parameter search
end
