function [A,B,C,D,Q,R,X,P,bestLogL,outLog,Pt]=EM(Y,U,Xguess,opts,Pguess,Ptguess)
%A true EM implementation to do LTI-SSM identification
%INPUT:
%Y is D2 x N
%U is D3 x N
%Xguess - Either the number of states for the system (if scalar) or a guess
%at the initial states of the system (if D1 x N matrix)

%Would randomly changing the scale every N iterations help in convergence speed? Maybe get unstuck from local saddles?

if nargin<4
    opts=[];
end
if nargin<5
    if ~isa(Y,'cell')
        Pguess=[];
    else
        Pguess=cell(size(Y));
    end
end
if nargin<6
    if ~isa(Y,'cell')
        Ptguess=[];
    else
        Ptguess=cell(size(Y));
    end
end
outLog=[];

%Process opts:
if isa(U,'cell')
  Nu=size(U{1},1);
  ny=size(Y{1},1);
else
  Nu=size(U,1);
  ny=size(Y,1);
end
if numel(Xguess)==1
  nx=numel(Xguess); %Order of model
else
  nx=size(Xguess,1);
end
[opts] = processEMopts(opts,Nu,nx,ny); %This is a fail-safe to check for proper options being defined.
if opts.fastFlag~=0 && ( (~isa(Y,'cell') && any(isnan(Y(:)))) || (isa(Y,'cell') && any(any(isnan(cell2mat(Y)))) ) )
   warning('EM:fastAndLoose','Requested fast filtering but data contains NaNs. Filtering/smoothing will be approximate, if it works at all. log-L is not guaranteed to be non-decreasing (disabling warning).')
   warning('off','EM:logLdrop') %If samples are NaN, fast filtering may make the log-L drop (smoothing is not exact, so the expectation step is not exact)
elseif opts.fastFlag~=0 && opts.fastFlag~=1
    warning('EM:fastFewSamples','Requested an exact number of samples for fast filtering. This is guaranteed to be equivalent to fast filtering only if the slowest time-constant of the system is much smaller than the requested number of samples, otherwise this is an appoximation.')
    warning('off','statKSfast:fewSamples')
end
if opts.logFlag
  outLog.opts=opts;
  tic
end
if opts.fastFlag~=0
    %Disable some annoying warnings related to fast filtering (otherwise these
    %warnings appear on each iteration when the Kalman filter is run):
    warning('off','statKFfast:unstable');
    warning('off','statKFfast:NaN');
    warning('off','statKSfast:unstable');
    warning('off','statKSfast:fewSamples');
end

%% ------------Init stuff:-------------------------------------------
% Init params
if ~iscell(Y)
    Yred=Y(opts.includeOutputIdx,:); %Does not work if it is a cell
else
    Yred=cellfun(@(x) x(opts.includeOutputIdx,:),Y,'UniformOutput',false);
end
[A1,B1,C1,D1,Q1,R1,x01,P01]=initEM(Yred,U,Xguess,opts,Pguess,Ptguess);
opts.evalLogL=true;
[X1,P1,Pt1,bestLogL]=estimateStates(Yred,A1,C1,Q1,R1,x01,P01,B1,D1,U,opts);


%Initialize log-likelihood register & current best solution:
logl=nan(opts.Niter,1); logl(1)=bestLogL;
A=A1; B=B1; C=C1; D=D1; Q=Q1; R=R1; x0=x01; P0=P01; P=P1; Pt=Pt1; X=X1;

%Initialize target logL:
if isempty(opts.targetLogL)
    opts.targetLogL=logl(1);
end
%% ----------------Now, do E-M-----------------------------------------
breakFlag=false;
msg='';
if opts.verbose
disp(['Iter = 1, target logL = ' num2str(opts.targetLogL,8) ', current logL=' num2str(bestLogL,8) ', \tau =' num2str(-1./log(sort(eig(A)))')])
end
dropCount=0;
opts.evalLogL=1; %Evaluating logL only once every N iterations. Shaves some computation time (~5%) when doing non-fast filtering
opts.dispStep=100; %Displaying info only once every M logL evluations
dS=opts.dispStep*opts.evalLogL;
for k=1:opts.Niter-1 %MAIN LOOP
	%E-step: compute the distribution of latent variables given current parameter estimates
	%M-step: find parameters A,B,C,D,Q,R that maximize likelihood of data

    %Save to log:
    if opts.logFlag
        outLog.vaps(k,:)=sort(eig(A1));
        outLog.logL=logl(k);
        outLog.runTime(k)=toc;
        tic;
    end

    %E-step:
    if mod(k,opts.evalLogL)==0; opts.noLogL=false; else; opts.noLogL=true; end
    [X1,P1,Pt1,l]=estimateStates(Yred,A1,C1,Q1,R1,x01,P01,B1,D1,U,opts);

    %Check stop conditions:
    [logl,dropCount,breakFlag,msg]=checkStopping(l,k,opts,logl,dropCount,bestLogL,X1);
    if l>bestLogL %There was improvement
      A=A1; B=B1; C=C1; D=D1; Q=Q1; R=R1; x0=x01; P0=P01; X=X1; P=P1; Pt=Pt1;
      bestLogL=l;
    end

    %Print some info
    if opts.verbose && (mod(k,dS)==0) %Print info
        lastChange=l-logl(k+1-dS); %Change since last printing
        if isnan(lastChange)
            1;
        end
        disp(['Iter = ' num2str(k) ', \Delta logL = ' num2str(lastChange,3) ', over target = ' num2str((l-opts.targetLogL),3) ', \tau =' num2str(-1./log(sort(eig(A1)))',3)]) %This displays logL over target, not in a per-sample per-dim way (easier to probe if logL is increasing significantly)
    end
    if breakFlag && ~opts.robustFlag
        break
    end
    
    %M-step:
    [A1,B1,C1,D1,Q1,R1,x01,P01]=estimateParams(Yred,U,X1,P1,Pt1,opts);

end %for loop

%% Some housekeeping
if opts.fastFlag~=0 %Re-enable disabled warnings
    warning('on','statKFfast:unstable');
    warning('on','statKFfast:NaN');
    warning('on','statKSfast:fewSamples');
    warning('on','statKSfast:unstable');
    warning('on','EM:logLdrop')
    warning('on','statKSfast:fewSamples')
%Comput optimal states and logL without the fastFlag
  opts.fastFlag=0;
  opts.noLogL=false;
  [X,P,Pt,bestLogL]=estimateStates(Yred,A1,C1,Q1,R1,x01,P01,B1,D1,U,opts);
end
if opts.logFlag
  outLog.vaps(k,:)=sort(eig(A1));
  outLog.runTime(k)=toc;
  outLog.breakFlag=breakFlag;
  outLog.msg=msg;
  outLog.bestLogL=bestLogL;
  %diary('off')
end
%%
%Display final message:
fprintf([msg ' \n']);
disp(['Finished. Number of iterations = ' num2str(k) ', logL = ' num2str(bestLogL,8) ', \tau =' num2str(-1./log(sort(eig(A)))')])

%% If some outputs were excluded, replace the corresponding values in C,R:
if size(Yred,1)<size(Y,1)
    Raux=R;
    R=diag(inf(ny,1));
    R(opts.includeOutputIdx,opts.includeOutputIdx)=Raux;
    Caux=C;
    C=zeros(size(C));
    C(opts.includeOutputIdx,:)=Caux;
    %And recompute the most likely values for D: (most likely values are not
    %the same with or without C contributions)
    Daux=D;
    D=Y/U;
    D(opts.includeOutputIdx,:)=Daux;
end
end  %Function

function [X1,P1,Pt1,l]=estimateStates(Y,A1,C1,Q1,R1,x01,P01,B1,D1,U,opts)
    %try %Because of precision issues, this may fail. The try-catch adds minor overhead
        if isa(Y,'cell') %Data is many realizations of same system
            [X1,P1,Pt1,~,~,~,~,~,l1]=cellfun(@(y,x0,p0,u) statKalmanSmoother(y,A1,C1,Q1,R1,x0,p0,B1,D1,u,opts),Y(:),x01(:),P01(:),U(:),'UniformOutput',false);
            l=sum(cell2mat(l1));
        else
            [X1,P1,Pt1,~,~,~,~,~,l]=statKalmanSmoother(Y,A1,C1,Q1,R1,x01,P01,B1,D1,U,opts);
        end
        %For future release: try a fast smoother, and if it fails or logl
        %drops, use the more accurate sqrt smoother. 
        %TO DO: covariance matrices from sqrt need to be converted to full
        %form (they are in the sqrt form)
    %catch %Slower but robust to non-psd matrices:
    %    disp('Smoothing failed, using sqrt-filter')
    %    if isa(Y,'cell') 
    %        [X1,P1,Pt1,~,~,~,l1]=cellfun(@(y,x0,p0,u) statSqrtSmoother(y,A1,C1,Q1,R1,x0,p0,B1,D1,u,opts),Y(:),x01(:),P01(:),U(:),'UniformOutput',false);
    %        l=sum(cell2mat(l1));
    %    else
    %        [X1,P1,Pt1,~,~,~,l]=statSqrtSmoother(Y,A1,C1,Q1,R1,x01,P01,B1,D1,U,opts);
    %    end
    %end
end

function [logl,dropCount,breakFlag,msg]=checkStopping(l,k,opts,logl,dropCount,bestLogL,Z)
    %There are three stopping criteria:
    %1) number of iterations
    %2) improvement in logL per dimension of output less than some threshold. It makes sense to do it per dimension of output because in high-dimensional models, the number of parameters of the model is roughly proportional to the number of output dimensions. IDeally, this would be done per number of free model parameters, so it has a direct link to significant improvements in log-L (Wilk's theorem suggests we should expect an increase in logL of 1 per each extra free param, so when improvement is well below this, we can stop).
    %3) relative improvement towards target value. The idea is that logL may be increasing fast according to criterion 2, but nowhere fast enough to ever reach the target value.
    breakFlag=false;
    msg='';
    if ~iscell(Z); Z={Z}; end
    if any(cellfun(@(x) any(imag(x(:))~=0),Z))
      msg='Complex states detected, stopping.';
      breakFlag=true;
    elseif any(cellfun(@(x) isnan(sum(x(:))),Z))
      msg='States are NaN, stopping.';
      breakFlag=true;
    end
    if mod(k,opts.evalLogL)==0 %Break conditions are only evaluated esporadically
      logl(k+1)=l;
      delta=l-logl(k);
      improvement=delta>=0;
      logL100ago=logl(max(k-100,1));
      targetRelImprovement100=(l-logL100ago)/(opts.targetLogL-logL100ago);
      belowTarget=max(l,bestLogL)<opts.targetLogL;
      relImprovementLast100=l-logL100ago; %Assessing the improvement on logl over the last 50 iterations (or less if there aren't as many)

      %Check for warning conditions:
      if ~improvement %This should never happen
          %Drops in logL may happen when using fast filtering in
          %conjunction with the presence of NaN samples, or with a fixed sample size. In that case, there
          %is no guarantee that the steady-state for the Kalman filter/smoother exists and is reached, and thus
          %filtering is approximate/not-optimal.
          %They may also happen when enforcing stable filtering (we are not really following EM then).
          %Finally, it may happen because of numerical issues: for example,
          %having a singular or close to singular, or ill-conditioned matrix R.
          if abs(delta)>1e-2
            %Report only if drops are larger than this. Arbitrary threshold.
            warning('EM:logLdrop',['logL decreased at iteration ' num2str(k) ', drop = ' num2str(delta)])
            dropCount=dropCount+1;
          end
      else
          dropCount=0; %Resetting drop counter
      end

      %Check for failure conditions:
      if imag(l)~=0 %This does not happen
          msg='Complex logL, probably ill-conditioned matrices involved. Stopping.';
          breakFlag=true;
      end
      %Check if we should stop early (to avoid wasting time):
      nny=length(opts.includeOutputIdx);
      if k>100 && (belowTarget && (targetRelImprovement100)<opts.targetTol) && ~opts.robustFlag%Breaking if improvement less than tol of distance to targetLogL
         msg='Unlikely to reach target value. Stopping.';
         breakFlag=true;
      elseif k>100 && (relImprovementLast100/nny)<opts.convergenceTol && ~opts.robustFlag
          %Considering the system stalled if improvement on logl per dimension of output is <tol
          msg='Increase is within tolerance (local max). Stopping.';
          breakFlag=true;
      elseif k==(opts.Niter-1)
          msg='Max number of iterations reached. Stopping.';
          breakFlag=true;
      elseif dropCount>10 %More than 10 drops in 100
          msg='log-L dropped 10 consecutive times. Possibly ill-conditioned solution. Stopping.';
          breakFlag=true;
      end
    end
end