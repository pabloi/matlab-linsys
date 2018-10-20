function [A,B,C,D,Q,R,X,P,bestLL,outLog]=randomStartEM(Y,U,nd,Nreps,opts)

%First iter:
fprintf(['\n Starting rep 0... \n']);
opts=processEMopts(opts);
outLog=struct();
opt1=opts;
opt1.fastFlag=true; %Enforcing fast filtering
[A,B,C,D,Q,R,X,P,bestLL,startLog]=EM(Y,U,nd,opt1);
try

    if opts.logFlag
        outLog.opts=opts;
        outLog.startLog=startLog;
        tic;
    end
catch
    bestLL=-Inf;
end
N=size(Y,2);
opts.targetLogL=bestLL;
for i=1:Nreps
    fprintf(['\n Starting rep ' num2str(i) '. Best logL so far=' num2str(bestLL,8) '... \n']);

    %Initialize starting point:
    x01=randn(nd,1);
    P01=1e1*eye(nd); %No sense in being certain about made up numbers
    %A1=diag(exp(-1./(.5*N*rand(nd,1)))); %WLOG, diagonal matrix with log-uniformly spaced time-constants
    A1=diag(exp(-1./exp(log(N)*rand(nd,1)))); %WLOG, diagonal matrix with log-uniformly spaced time-constants in the [1,N] interval
    %I think the sign above is unnecessary
    B1=ones(nd,size(U,1)); %WLOG
    Q1=(abs(randn)+1e-4)*eye(nd); %Needs to be psd
    [~,Xsmooth]=fwdSim(U,A1,B1,zeros(1,nd),0,[],Q1,[]);
    C1=randn(size(Y,1),nd)/size(Y,1); %WLOG
    D1=randn(size(Y,1),size(U,1));
    %Alt C,D:
    %CD1=Y/[Xsmooth(:,1:end-1);U];
    %C1=CD1(:,1:nd);
    %D1=CD1(:,nd+1:end);
    z=Y-C1*Xsmooth(:,1:end-1)-D1*U;
    idx=~any(isnan(z));
    z=z(:,idx);
    R1=z*z'/size(z,2) + C1*Q1*C1'; %Reasonable estimate of R
    %Alt R:
    %R1=(abs(randn)+1e-4)*eye(size(Y,1)); %Needs to be psd
    [Xguess]=statKalmanSmoother(Y,A1,C1,Q1,R1,x01,P01,B1,D1,U,[],0);
    Xguess=medfilt1(Xguess,9,[],2); %Some smoothing to avoid starting with very ugly estimates

    %Optimize:
    try %Sometimes ill conditioned problems fail
      [Ai,Bi,Ci,Di,Qi,Ri,Xi,Pi,logl,repLog]=EM(Y,U,Xguess,opts);
      if opts.logFlag
          outLog.repLog{k}=repLog;
          outLog.repRunTime(k)=toc;
          tic;
      end
      %If solution improved, save and display:
      if logl>bestLL
          A=Ai; B=Bi; C=Ci; D=Di; Q=Qi; R=Ri; X=Xi; P=Pi;
          bestLL=logl;
          opts.targetLogL=bestLL;
          disp(['Success, best logL=' num2str(bestLL,8)])
      end
    catch
      1;
    end

end

disp(['Refining solution...']);
opts.Niter=3e4;
opts.convergenceTol=1e-11;
opts.targetTol=0;
try
    [Ai,Bi,Ci,Di,Qi,Ri,Xi,Pi,bestLL1,refineLog]=EM(Y,U,X,opts,P); %Refine solution, sometimes works
catch
    bestLL1=bestLL;
end
if bestLL1>bestLL
    A=Ai; B=Bi; C=Ci; D=Di; Q=Qi; R=Ri; X=Xi; P=Pi; bestLL=bestLL1;
end

disp(['End. Best logL=' num2str(bestLL,8)]);
if opts.logFlag
    outLog.repfineLog=refineLog;
    outLog.refineRunTime=toc;
end
end
