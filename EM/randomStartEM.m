function [A,B,C,D,Q,R,X,P,bestLL,outLog]=randomStartEM(Y,U,nd,opts)

%First iter:
fprintf(['\n Starting rep 0 (short one)... \n']);
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

opt1=opts; %Options for rapid round
opt2=opts; %Options for other rounds
optR=processEMopts(opts,Nu,nd,ny); %Options for using within this function, never passed elsewhere

outLog=struct();
%opt1.fixP0=diag(Inf(nd,1));
opt1.Niter=100; %Very fast evaluation of initial case, just to get a benchmark.
warning('off','EM:logLdrop') %If samples are NaN, fast filtering may make the log-L drop (smoothing is not exact, so the expectation step is not exact)
warning('off','EM:fastAndLoose')%Disabling the warning that NaN and fast may be happening
[A,B,C,D,Q,R,X,P,bestLL,startLog,Pt]=EM(Y,U,nd,opt1);
warning('on','EM:logLdrop')
warning('on','EM:fastAndLoose')
if optR.logFlag
    outLog.opts=opts;
    outLog.startLog=startLog;
    tic;
end
opt2.targetLogL=bestLL;

lastSuccess=0;
warning('off','statKF:logLnoPrior'); %Enforcing improper prior for parameter search
for i=1:optR.Nreps
    fprintf(['\n Starting rep ' num2str(i) '. Best logL so far=' num2str(bestLL,8) ' (iter=' num2str(lastSuccess) ') \n']);

    %Initialize starting point:
    Xguess=guess(nd,Y,U,optR);

    %Optimize:
    %try
	[Ai,Bi,Ci,Di,Qi,Ri,Xi,Pi,logl,repLog,Pti]=EM(Y,U,Xguess,opt2);

    %If solution improved, save and display:
    if logl>bestLL
        A=Ai; B=Bi; C=Ci; D=Di; Q=Qi; R=Ri; X=Xi; P=Pi; Pt=Pti;
        bestLL=logl;            opt2.targetLogL=bestLL;
        lastSuccess=i;
        disp(['----Success, best logL=' num2str(bestLL,8) '(iter=' num2str(lastSuccess) ')----'])
    end
    if optR.logFlag
        outLog.repLog{i}=repLog;  outLog.repRunTime(i)=toc;   tic;
    end

    %catch ME
	%warning('randomStartEM:EMiterFail',['Iteration #' num2str(i) ' failed with error ' ME.identifier '. This could be a numerical issue or something more important. Please look at previous messages. Ignoring and continuing'])
    %end
end
warning('on','statKF:logLnoPrior');

refineLog=[];
if ~optR.disableRefine && optR.refineFastFlag && optR.fastFlag~=0 %Fast allowed
  disp(['Refining solution... (fast) Best logL so far=' num2str(bestLL,8) '(iter=' num2str(lastSuccess) ')']);
  opt2.Niter=optR.refineMaxIter; %This will go fast, can afford to have many iterations, it will rarely reach the limit.
  opt2.convergenceTol=optR.refineTol/optR.fastRefineTolFactor;
  opt2.targetTol=1e-4;
  opt2.fastFlag=50;
  opt2.targetLogL=bestLL;
  [Ai,Bi,Ci,Di,Qi,Ri,Xi,Pi,bestLL1,refineLog,Pti]=EM(Y,U,X,opt2,P,Pt);
  if bestLL1>bestLL
      A=Ai; B=Bi; C=Ci; D=Di; Q=Qi; R=Ri; X=Xi; P=Pi; bestLL=bestLL1; Pt=Pti;
    else
      warning('Fast refining did not work (?)')
      %This can happen if we have NaN samples, as the fast filtering may be crap
      %In general, fast filtering with NaN samples is NOT encouraged.
  end
end

disp(['Refining solution... (patient mode) Best logL so far=' num2str(bestLL,8)]);
opt2.fastFlag=0;
opt2.Niter=optR.refineMaxIter;
opt2.convergenceTol=optR.refineTol;
opt2.targetLogL=bestLL;
if ~optR.disableRefine
[Ai,Bi,Ci,Di,Qi,Ri,Xi,Pi,bestLL1,refineLog2,Pti]=EM(Y,U,X,opt2,P,Pt); %Refine solution, should work always
if bestLL1>bestLL
    A=Ai; B=Bi; C=Ci; D=Di; Q=Qi; R=Ri; X=Xi; P=Pi; bestLL=bestLL1; Pt=Pti;
end
end

disp(['End. Best logL=' num2str(bestLL,8)]);
if optR.logFlag
    outLog.refineLog=refineLog;
    outLog.refineLog2=refineLog2;
    outLog.refineRunTime=toc;
end
end

function Xguess=guess(nd,Y,U,opts)
    if isa(U,'cell')
        u=cell2mat(U(:)');
        y=cell2mat(Y(:)');
    else
        y=Y;
        u=U;
    end
    y=y(opts.includeOutputIdx,:);
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
      idx=~any(isnan(z),1);
      z=z(:,idx);
      R1=z*z'/size(z,2) + C1*Q1*C1'; %Reasonable estimate of R
    else
      R1=opts.fixR;
    end
    warning('off','statKF:logLnoPrior') %Using uninformative prior
    warning('off','statKSfast:fewSamples') %If using fast mode, it doesnt matter here
    [Xguess]=statKalmanSmoother(y,A1,C1,Q1,R1,x0,P0,B1,D1,u,opts);
    Xguess(isnan(Xguess))=0; %Doxy. a better workaround is needed
    warning('on','statKF:logLnoPrior')
    warning('on','statKSfast:fewSamples')
    %Alternative: [Xguess]=statInfoSmoother2(y,A1,C1,Q1,R1,[],[],B1,D1,u,opts);
    Xguess=medfilt1(Xguess,9,[],2); %Some smoothing to avoid starting with very ugly estimates
    if isa(U,'cell')
        Xguess=mat2cell(Xguess,size(Xguess,1),cellfun(@(x) size(x,2),Y));
    end
end
