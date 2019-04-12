function [opts] = processEMopts(opts,nu,nx,ny)

%Define default values:
defaultOpts.Niter=2e3;%Max number of iters
defaultOpts.Nreps=10;%Number of repetitions for randomStartEM
defaultOpts.robustFlag=false;
defaultOpts.outlierReject=false; %No rejection of outliers in KF/KS
defaultOpts.fastFlag=1;  %statKF/KS auto-select fast samples by default
defaultOpts.convergenceTol=5e-3; % 5e-3 minimum improvement in logL (per dim, but not per sample) every 100  iterations
defaultOpts.targetTol=1e-3;  % .1% minimum improvement towards target every 100 iters. This is possibly TOO tolerant, it never stops an iteration early, no matter what.
defaultOpts.targetLogL=[];
defaultOpts.diagA=false;
defaultOpts.sphericalR=false;
defaultOpts.diagR=false;
defaultOpts.thR=0;
defaultOpts.outlierReject=false;
defaultOpts.indD=1:nu; %Include all
defaultOpts.indB=1:nu; %Include all
defaultOpts.logFlag=false; %No log-keeping by default
defaultOpts.fixA=[];
defaultOpts.fixB=[];
defaultOpts.fixC=[];
defaultOpts.fixD=[];
defaultOpts.fixQ=[];
defaultOpts.fixR=[];
defaultOpts.fixX0=[];
defaultOpts.fixP0=[];
defaultOpts.includeOutputIdx=1:ny; %Include all
defaultOpts.stableA=false; %Not enforcing stability of A by default

%Options for randomStartEM's final refinement stage (iterated EM)
defaultOpts.refineTol=1e-4; %Very picky, increase in logL every 1e2 iterations
%Implies that in 1e4 iterations the logL will increase 1e-2 at least.
%For high dimensional output with a single state, this is a sensible, if conservative, choice
%as it limits iterations when is too slow with respect to Wilk's logL overfit
%limiting theorem. For mulitple states we are being conservative
%(more free parameters means that logL should increase even more to be signficant).
defaultOpts.refineMaxIter=2e4;
defaultOpts.refineFastFlag=false; %Patient mode

%Assign any options that were not provided with default values:
fNames=fieldnames(defaultOpts);
for i=1:length(fNames)
  if ~isfield(opts,fNames{i}) || isempty(opts.(fNames{i}))
    opts.(fNames{i})=defaultOpts.(fNames{i});
  end
end

%Do a sanity check on options that may be conflicting:
if ~isempty(opts.fixB)
  opts.indB=1:nu;
  if any(size(opts.fixB)~=[nx,nu])
    error('EMopts:providedBdimMismatch','Provided B matrix size is inconsistent with number of inputs or states.')
  end
end
if ~isempty(opts.fixD)
  opts.indD=1:nu;
  if any(size(opts.fixD)~=[ny,nu])
    error('EMopts:providedDdimMismatch','Provided D matrix size is inconsistent with number of inputs or outputs.')
  end
end
if islogical(opts.includeOutputIdx) %Logical indexing used
    opts.includeOutputIdx=find(opts.includeOutputIdx);
end

%Check sizes and shrink fixC, fixD, fixR if needed:
nny=numel(opts.includeOutputIdx);
if ~isempty(opts.fixC)
    if size(opts.fixC,2)~=nx || ~any(size(opts.fixC,1)==[ny,nny])
        error('processEMopts:fixCsize','Incorrect fixed C matrix size.');
    elseif size(opts.fixC,1)==ny
        opts.fixC=opts.fixC(opts.includeOutputIdx,:);
    elseif size(opts.fixC,1)==nny %Presuming already shrunk, do issue warning
        warning('processEMopts:fixCsize','Size of fixed C was not as expected. It appears to be a reduced version, which is not default behavior. Proceeding, but please fix.')
    end
end
if ~isempty(opts.fixD)
    if size(opts.fixD,2)~=nu || ~any(size(opts.fixD,1)==[ny,nny])
        error('processEMopts:fixDsize','Incorrect fixed D matrix size.');
    elseif size(opts.fixD,1)==ny
        opts.fixD=opts.fixD(opts.includeOutputIdx,:);
    elseif size(opts.fixD,1)==nny %Presuming already shrunk, do issue warning
        warning('processEMopts:fixDsize','Size of fixed D was not as expected. It appears to be a reduced version, which is not default behavior. Proceeding, but please fix.')
    end
end
if ~isempty(opts.fixR)
    if size(opts.fixR,2)~=size(opts.fixR,1) || ~any(size(opts.fixR,1)==[ny,nny])
        error('processEMopts:fixRsize','Incorrect fixed R matrix size. Should be square and of same size as output.');
    elseif size(opts.fixR,1)==ny
        opts.fixR=opts.fixR(opts.includeOutputIdx,opts.includeOutputIdx);
    elseif size(opts.fixR,1)==nny %Presuming already shrunk, do issue warning
        warning('processEMopts:fixRsize','Size of fixed R was not as expected. It appears to be a reduced version, which is not default behavior. Proceeding, but please fix.')
    end
end

%Reintepret some special options:
if isnan(opts.fixQ)
  opts.fixQ=zeros(nx);
end
if isnan(opts.fixX0)
  opts.fixX0=zeros(nx,1);
  opts.fixP0=zeros(nx); %No uncertainty if initial state is given
end
if isnan(opts.fixP0)
  opts.fixP0=diag(inf(size(opts.fixX0))); %Max uncertainty
end
if islogical(opts.includeOutputIdx)
  if length(opts.includeOutputIdx)~=ny
    error('EMopts:outputIdxListSizeMismatch','Provided list of output indexes to be included is inconsistent with output size.')
  else
  opts.includeOutputIdx=find(opts.includeOutputIdx); %Logical to index list
  end
end
