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
    parForArg=pp.NumWorkers;
    innerLoopSize=pp.NumWorkers;
    outerLoopSize=ceil(Nreps/innerLoopSize);
end

%First iter:
fprintf(['Starting rep 0... ']);
[A,B,C,D,Q,R,X,P]=EM(Y,U,nd,[],1); %Fast EM is used for the first iter
startLL=dataLogLikelihood(Y,U,A,B,C,D,Q,R,X(:,1),P(:,:,1));
bestLL=startLL;
logl=nan(1,outerLoopSize);
m=struct();

spmd (parForArg)
    for i=1:outerLoopSize %Each worker will go through this loop
        fprintf(['Starting rep ' num2str(i+(labindex-1)*outerLoopSize) '. Best logL so far=' num2str(bestLL) '... ']);

        %Initialize starting point:
        x01=randn(nd,1);
        P01=1e5*eye(nd); %No sense in being certain about made up numbers
        A1=diag(rand(nd,1)); %WLOG
        B1=randn(nd,size(U,1)); 
        C1=randn(size(Y,1),nd)/size(Y,1); %WLOG
        D1=randn(size(Y,1),1);
        Q1=(abs(randn)+1e-7)*eye(nd); %Needs to be psd
        R1=(abs(randn)+1e-7)*eye(size(Y,1)); %Needs to be psd
        [Xguess]=statKalmanSmoother(Y,A1,C1,Q1,R1,x01,P01,B1,D1,U);
        Xguess=medfilt1(Xguess,9,[],2);
        
        %Optimize:
        [Ai,Bi,Ci,Di,Qi,Ri,Xi,Pi]=trueEM(Y,U,Xguess,bestLL,fastFlag);

        %Save results:
        logl(i)=dataLogLikelihood(Y,U,Ai,Bi,Ci,Di,Qi,Ri,Xi(:,1),Pi(:,:,1));
        
        %Get new current best if available:
        if parForArg>0 && labProbe
            bestLL=labReceive;
        end
        if logl(i)>bestLL
            disp(['Success, best logL=' num2str(logl(i))])
            bestLL=logl(i);
            m.A=Ai;
            m.B=Bi;
            m.C=Ci;
            m.D=Di;
            m.Q=Qi;
            m.R=Ri;
            m.X=Xi;
            m.P=Pi;
            if parForArg>0
                labBroadcast(labindex,bestLL);
            end
        end
    end
end

%Select best solution:
logl=cell2mat(logl(:));
logl=max(logl,[],2); %Best logl for each parallel worker
[~,idx]=max(logl); %Best across workers
m=m{idx};

if logl(idx)>startLL %Best model is better than original one
    startLL=logl(idx);
    A=m.A;
    B=m.B;
    C=m.C;
    D=m.D;
    Q=m.Q;
    R=m.R;
    X=m.X;
    P=m.P;
end
disp(['End. Best logL=' num2str(startLL)]);

end
