function [A,B,C,D,Q,R,X,P]=randomStartEM_par(Y,U,nd,Nreps,method)

if nargin<4 || isempty(Nreps)
    Nreps=20;
end

if nargin<5 || isempty(method)
   method='true'; 
end %TODO: if method is given, check that it is 'true' or 'fast'
switch method
    case 'true'
        fastFlag=0;
    otherwise
        fastFlag=1;
end

pp=gcp('nocreate');
if isempty(pp) %No parallel pool open, doing regular for, issue warning
    warning('No parallel pool was found, running for loop in serial way.')
    %poolFlag=true;
    parForArg=0; %Argument needed to run parfor serially
    innerLoopSize=1;
    outerLoopSize=Nreps;
else
    parForArg=Inf;
    innerLoopSize=min(Nreps,1*pp.NumWorkers);
    outerLoopSize=ceil(Nreps/innerLoopSize);
end

%First iter:
[A,B,C,D,Q,R,X,P]=trueEM(Y,U,nd,[],1); %Fast EM is used for the first iter
bestLL=dataLogLikelihood(Y,U,A,B,C,D,Q,R,X(:,1),P(:,:,1));
logl=nan(innerLoopSize,outerLoopSize);
m=cell(innerLoopSize,outerLoopSize);

for j=1:outerLoopSize %Outer loop is necessary to update bestLL
    parfor (i=1:innerLoopSize, parForArg)
        disp(['Starting rep ' num2str(i+(j-1)*innerLoopSize) '. Best logL so far=' num2str(bestLL)]);

        %Initialize starting point:
        x01=randn(nd,1);
        P01=1e5*eye(nd); %No sense in being certain about made up numbers
        A1=diag(rand(nd,1)); %WLOG
        B1=randn(nd,size(U,1)); 
        C1=randn(size(Y,1),nd)/size(Y,1); %WLOG
        D1=randn(size(Y,1),1);
        Q1=(abs(randn)+1e-5)*eye(nd); %Needs to be psd
        R1=(abs(randn)+1e-5)*eye(size(Y,1)); %Needs to be psd
        [Xguess]=statKalmanSmoother(Y,A1,C1,Q1,R1,x01,P01,B1,D1,U);
        Xguess=medfilt1(Xguess,9,[],2);
        
        %Optimize:
        [Ai,Bi,Ci,Di,Qi,Ri,Xi,Pi]=trueEM(Y,U,Xguess,bestLL,fastFlag);

        %Save results:
        m{i,j}.A=Ai;
        m{i,j}.B=Bi;
        m{i,j}.C=Ci;
        m{i,j}.D=Di;
        m{i,j}.Q=Qi;
        m{i,j}.R=Ri;
        m{i,j}.X=Xi;
        m{i,j}.P=Pi;
        logl(i,j)=dataLogLikelihood(Y,U,Ai,Bi,Ci,Di,Qi,Ri,Xi(:,1),Pi(:,:,1));
        if logl(i,j)>bestLL
            disp(['Success, best logL=' num2str(logl(i,j))])
        end
    end
    
    %In the outer-loop only: update the best logL() so far
    if max(logl(:))>bestLL
        [~,idx]=max(logl(:));
        bestLL=logl(idx);
        disp(['Inner loop done, updating best logL=' num2str(logl(idx))])
    end
end

%Select best solution:
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

end