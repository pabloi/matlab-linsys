function [A,B,C,D,Q,R,X,P]=randomStartEM_par(Y,U,nd,Nreps,method)
if nargin<4 || isempty(Nreps)
    Nreps=20;
end
if nargin<5 || isempty(method)
   method='true'; 
end %TODO: if method is given, check that it is 'true' or 'fast'

poolFlag=false;
if isempty(gcp('nocreate')) %No parallel pool open
    poolFlag=true;
end
pp=gcp;
innerLoopSize=min(Nreps,3*pp.NumWorkers);
outerLoopSize=ceil(Nreps/innerLoopSize);
Nreps=outerLoopSize*innerLoopSize;

[A,B,C,D,Q,R,X,P]=trueEM(Y,U,nd);
bestLL=dataLogLikelihood(Y,U,A,B,C,D,Q,R,X(:,1),P(:,:,1));
logl=nan(innerLoopSize,outerLoopSize);
m=cell(innerLoopSize,outerLoopSize);

for j=1:outerLoopSize %Outer loop is necessary to update bestLL
    parfor i=1:innerLoopSize
        disp(['Rep ' num2str(i+(j-1)*innerLoopSize)]);
        %Xguess=randn(D1,N);
        x01=randn(nd,1);
        P01=randn(nd,nd);
        A1=randn(nd,nd);
        B1=ones(nd,size(U,1)); %WLOG
        C1=randn(size(Y,1),nd);
        D1=randn(size(Y,1),1);
        Q1=randn*eye(nd);
        R1=randn*eye(size(Y,1));
        [Xguess]=statKalmanSmoother(Y,A1,C1,Q1,R1,x01,P01,B1,D1,U);
        Xguess=medfilt1(Xguess,9,[],2);
        switch method
            case 'fast'
                [Ai,Bi,Ci,Di,Qi,Ri,Xi,Pi]=fastEM(Y,U,Xguess);
            case 'true'
                [Ai,Bi,Ci,Di,Qi,Ri,Xi,Pi]=trueEM(Y,U,Xguess,bestLL);
        end
        m{i,j}.A=Ai;
        m{i,j}.B=Bi;
        m{i,j}.C=Ci;
        m{i,j}.D=Di;
        m{i,j}.Q=Qi;
        m{i,j}.R=Ri;
        m{i,j}.X=Xi;
        m{i,j}.P=Pi;
        logl(i,j)=dataLogLikelihood(Y,U,Ai,Bi,Ci,Di,Qi,Ri,Xi(:,1),Pi(:,:,1));
    end
    if max(logl)>bestLL
        [~,idx]=max(logl(:));
        bestLL=logl(idx);
        disp(['Success, best logL=' num2str(logl(idx))])
    end
end
[~,idx]=max(logl(:));
disp(['End. Best logL=' num2str(logl(idx))]);
A=m{idx}.A;
B=m{idx}.B;
C=m{idx}.C;
D=m{idx}.D;
Q=m{idx}.Q;
R=m{idx}.R;
X=m{idx}.X;
P=m{idx}.P;
if poolFlag
    pp.delete
end
end