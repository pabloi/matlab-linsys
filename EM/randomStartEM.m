function [A,B,C,D,Q,R,X,P,bestLL,outLog]=randomStartEM(Y,U,nd,opts)

%First iter:
fprintf(['\n Starting rep 0 (fast one)... \n']);
%Pre-process optional flags:
if isa(U,'cell')
  Nu=size(U{1},1);
else
  Nu=size(U,1);
end
opts=processEMopts(opts,Nu);
outLog=struct();
opt1=opts;
opt1.fastFlag=true; %Enforcing fast filtering
opt1.Niter=min([opt1.Niter,500]); %Very fast evaluation of initial case, just to get a benchmark.
warning('off','EM:logLdrop') %If samples are NaN, fast filtering may make the log-L drop (smoothing is not exact, so the expectation step is not exact)
warning('off','EM:fastAndLoose')%Disabling the warning that NaN and fast may be happening
[A,B,C,D,Q,R,X,P,bestLL,startLog]=EM(Y,U,nd,opt1);
warning('on','EM:logLdrop')
warning('on','EM:fastAndLoose')
if opts.logFlag
    outLog.opts=opts;
    outLog.startLog=startLog;
    tic;
end
opts.targetLogL=bestLL;
lastSuccess=0;
for i=1:opts.Nreps
    fprintf(['\n Starting rep ' num2str(i) '. Best logL so far=' num2str(bestLL,8) ' (iter=' num2str(lastSuccess) ') \n']);

    %Initialize starting point:
    Xguess=guess(nd,Y,U,opts);

    %Optimize:
    [Ai,Bi,Ci,Di,Qi,Ri,Xi,Pi,logl,repLog]=EM(Y,U,Xguess,opts);

    %If solution improved, save and display:
      if logl>bestLL
          A=Ai; B=Bi; C=Ci; D=Di; Q=Qi; R=Ri; X=Xi; P=Pi;
          bestLL=logl;            opts.targetLogL=bestLL;
          lastSuccess=i;
          disp('--')
          disp('--')
          disp(['Success, best logL=' num2str(bestLL,8) '(iter=' num2str(lastSuccess) ')'])
          disp('--')
          disp('--')
      end
      if opts.logFlag
          outLog.repLog{i}=repLog;  outLog.repRunTime(i)=toc;   tic;
      end
end

disp(['Refining solution... Best logL so far=' num2str(bestLL,8) '(iter=' num2str(lastSuccess) ')']);
opts.Niter=2e4;
opts.convergenceTol=5e-4; %This is in logL per output dimension every 1e2 iterations (see EM). Implies that in 1e4 iterations the logL will increase 5e-2 at least. For high dimensional output with a single state, this is a sensible choice as it limits iterations when is too slow with respect to Wilk's logL overfit limiting theorem. For mulitple states we are being conservative (more free parameters means that logL should increase even more to be signficant).
opts.targetTol=0;
opts.fastFlag=false; %Patience
[Ai,Bi,Ci,Di,Qi,Ri,Xi,Pi,bestLL1,refineLog]=EM(Y,U,X,opts,P); %Refine solution, sometimes works
if bestLL1>bestLL
    A=Ai; B=Bi; C=Ci; D=Di; Q=Qi; R=Ri; X=Xi; P=Pi; bestLL=bestLL1;
end

disp(['End. Best logL=' num2str(bestLL,8)]);
if opts.logFlag
    outLog.repfineLog=refineLog;
    outLog.refineRunTime=toc;
end
end

function Xguess=guess(nd,Y,U,opts)
    if isa(U,'cell')
        u=cell2mat(U);
        y=cell2mat(Y);
    else
        y=Y;
        u=U;
    end
    [ny,N]=size(y);
    if isempty(opts.fixA)
      A1=diag(exp(-1./exp(log(N)*rand(nd,1)))); %WLOG, diagonal matrix with log-uniformly spaced time-constants in the [1,N] interval
        %I think the sign above is unnecessary
    else
      A1=opts.fixA;
    end
    if isempty(opts.fixB)
      B1=ones(nd,size(u,1)); %WLOG
    else
      B1=opts.fixB;
    end
    if isempty(opts.fixQ)
      Q1=(abs(randn)+1e-4)*eye(nd); %Needs to be psd
    else
      Q1=opts.fixQ;
    end
    if isempty(opts.fixC)
      C1=randn(ny,nd)/ny; %WLOG
    else
      C1=opts.fixC;
    end
    if isempty(opts.fixD)
      D1=randn(ny,size(u,1));
    else
      D1=opts.fixD;
    end
    x0=opts.fixX0; %These are empty by default
    P0=opts.fixP0;
    [~,Xsmooth]=fwdSim(u,A1,B1,zeros(1,nd),zeros(1,size(u,1)),x0,Q1,[]);
    if isempty(opts.fixR)
      z=y-C1*Xsmooth(:,1:end-1)-D1*u;
      idx=~any(isnan(z));
      z=z(:,idx);
      R1=z*z'/size(z,2) + C1*Q1*C1'; %Reasonable estimate of R
    else
      R1=opts.fixR;
    end
    warning('off','statKF:logLnoPrior') %Using uninformative prior
    [Xguess]=statKalmanSmoother(y,A1,C1,Q1,R1,x0,P0,B1,D1,u,opts);
    warning('on','statKF:logLnoPrior')
    %Alternative: [Xguess]=statInfoSmoother2(y,A1,C1,Q1,R1,[],[],B1,D1,u,opts);
    Xguess=medfilt1(Xguess,9,[],2); %Some smoothing to avoid starting with very ugly estimates
    if isa(U,'cell')
        Xguess=mat2cell(Xguess,size(Xguess,1),cellfun(@(x) size(x,2),Y));
    end
end
