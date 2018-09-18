function [A,B,C,D,Q,R,X,P,bestLL]=randomStartEM(Y,U,nd,Nreps,method)
if nargin<4 || isempty(Nreps)
    Nreps=20;
end
if nargin<5 || isempty(method)
   method='true'; 
end %TODO: if method is given, check that it is 'true' or 'fast'

%First iter:
fprintf(['Starting rep 0... ']);
Niter=301;
[A,B,C,D,Q,R,X,P,bestLL]=EM(Y,U,nd,[],0,[],Niter); %FastEM
N=size(Y,2);

switch method
    case 'true'
        fastFlag=[];
    otherwise
        fastFlag=0;
end

for i=1:Nreps
    fprintf(['Starting rep ' num2str(i) '. Best logL so far=' num2str(bestLL) '... ']);

    %Initialize starting point:
    x01=randn(nd,1);
    P01=1e5*eye(nd); %No sense in being certain about made up numbers
    A1=diag(rand(nd,1).*sign(randn(nd,1))); %WLOG
    B1=randn(nd,size(U,1));
    C1=randn(size(Y,1),nd)/size(Y,1); %WLOG
    D1=randn(size(Y,1),1);
    Q1=(abs(randn)+1e-7)*eye(nd); %Needs to be psd
    R1=(abs(randn)+1e-7)*eye(size(Y,1)); %Needs to be psd
    [Xguess]=statKalmanSmoother(Y,A1,C1,Q1,R1,x01,P01,B1,D1,U,[],0);
    Xguess=medfilt1(Xguess,9,[],2);

    %Optimize:
    [Ai,Bi,Ci,Di,Qi,Ri,Xi,Pi,logl]=EM(Y,U,Xguess,bestLL,fastFlag,[],Niter);
    %logl=dataLogLikelihood(Y,U,Ai,Bi,Ci,Di,Qi,Ri,Xi(:,1),Pi(:,:,1));

    %If solution improved, save and display:
    if logl>bestLL
        A=Ai; B=Bi; C=Ci; D=Di; Q=Qi; R=Ri; X=Xi; P=Pi;
        bestLL=logl;
        disp(['Success, best logL=' num2str(bestLL)])
    end
end
disp(['End. Best logL=' num2str(bestLL)]);

[Ai,Bi,Ci,Di,Qi,Ri,Xi,Pi,bestLL1]=EM(Y,U,X,bestLL,[],[],200); %Refine solution, sometimes works
if bestLL1>bestLL
    A=Ai; B=Bi; C=Ci; D=Di; Q=Qi; R=Ri; X=Xi; P=Pi; bestLL=bestLL1;
end
    
end
