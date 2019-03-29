function [A,B,C,D,Q,R,X,P,bestLL,outLog]=randomStartEM(Y,U,nd,opts)

%First iter:
fprintf(['\n Starting rep 0 (fast one)... \n']);
%Pre-process optional flags:
if isa(U,'cell')
  Nu=size(U{1},1);
  ny=size(Y{1},1);
  Nsamp=max(cellfun(@(x) size(x,2),Y));
else
  Nu=size(U,1);
  ny=size(Y,1);
  Nsamp=size(Y,2);
end
opts=processEMopts(opts,Nu,nd,ny);
outLog=struct();
opt1=opts;
opt1.fastFlag=50; %Enforcing fast filtering
%opt1.fixP0=diag(Inf(nd,1));
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
opt2=opts;
%if opt2.fastFlag~=0
%  opt2.convergenceTol=opt2.convergenceTol/10; %In fast mode it seems more likely to find flat-ish regions, and since things go faster, there is little damage doing this.
%end
%opt2.fixP0=diag(Inf(nd,1));
lastSuccess=0;
warning('off','statKF:logLnoPrior'); %Enforcing improper prior for parameter search
for i=1:opts.Nreps
    fprintf(['\n Starting rep ' num2str(i) '. Best logL so far=' num2str(bestLL,8) ' (iter=' num2str(lastSuccess) ') \n']);

    %Initialize starting point:
    Xguess=guess(nd,Y,U,opts);

    %Optimize:
    [Ai,Bi,Ci,Di,Qi,Ri,Xi,Pi,logl,repLog]=EM(Y,U,Xguess,opt2);

    %If solution improved, save and display:
    if logl>bestLL
        A=Ai; B=Bi; C=Ci; D=Di; Q=Qi; R=Ri; X=Xi; P=Pi;
        bestLL=logl;            opt2.targetLogL=bestLL;
        lastSuccess=i;
        disp(['----Success, best logL=' num2str(bestLL,8) '(iter=' num2str(lastSuccess) ')----'])
    end
    if opts.logFlag
        outLog.repLog{i}=repLog;  outLog.repRunTime(i)=toc;   tic;
    end
end
warning('on','statKF:logLnoPrior');

if opts.fastFlag~=0 %Fast allowed
  disp(['Refining solution... (fast) Best logL so far=' num2str(bestLL,8) '(iter=' num2str(lastSuccess) ')']);
  opts.Niter=opts.refineMaxIter; %This will go fast, can afford to have many iterations, it will rarely reach the limit.
  opts.convergenceTol=opts.refineTol/1e4; %This is mostly to prevent the algorithm from stopping at a flat-ish region
  opts.targetTol=1e-4;
  opts.fastFlag=50;
  opts.targetLogL=bestLL;
  [Ai,Bi,Ci,Di,Qi,Ri,Xi,Pi,bestLL1,refineLog]=EM(Y,U,X,opts,P);
  if bestLL1>bestLL
      A=Ai; B=Bi; C=Ci; D=Di; Q=Qi; R=Ri; X=Xi; P=Pi; bestLL=bestLL1;
    else
      warning('Fast refining did not work (?)')
      %This can happen if we have NaN samples, as the fast filtering may be crap
      %In general, fast filtering with NaN samples is NOT encouraged.
  end
end

disp(['Refining solution... (patient mode) Best logL so far=' num2str(bestLL,8)]);
opts.fastFlag=0;
opts.Niter=opts.refineMaxIter;
opts.convergenceTol=opts.refineTol;
opts.targetLogL=bestLL;
[Ai,Bi,Ci,Di,Qi,Ri,Xi,Pi,bestLL1,refineLog2]=EM(Y,U,X,opts,P); %Refine solution, should work always
if bestLL1>bestLL
    A=Ai; B=Bi; C=Ci; D=Di; Q=Qi; R=Ri; X=Xi; P=Pi; bestLL=bestLL1;
end

disp(['End. Best logL=' num2str(bestLL,8)]);
if opts.logFlag
    outLog.repfineLog=refineLog;
    outLog.repfineLog2=refineLog2;
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
      A1=diag(exp(-1./exp(log(N/2)*rand(nd,1)))); %WLOG, diagonal matrix with log-uniformly spaced time-constants in the [1,N/2] interval
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
    elseif ~isnan(opts.fixQ)
      Q1=opts.fixQ;
    else
      Q1=zeros(size(A1));
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
