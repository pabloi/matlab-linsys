function [opts] = processEMopts(opts,nu)

if ~isfield(opts,'Niter')
    opts.Niter=1e3; %Max number of iters
end

if ~isfield(opts,'robustFlag')
    opts.robustFlag=false; % Non-robust behavior
end

if ~isfield(opts,'fastFlag')
    opts.fastFlag=1; %statKF/KS auto-select fast samples by default
end

if ~isfield(opts,'convergenceTol')
    opts.convergenceTol=1e-7; % 1e-7 minimum improvement in logL every 50 strides
end

if ~isfield(opts,'targetTol')
    opts.targetTol=1e-3; % .1% minimum improvement towards target every 50 iters
end

if ~isfield(opts,'targetLogL')
    opts.targetLogL=[];
end

if ~isfield(opts,'sphericalR')
    opts.sphericalR=false;
end
if ~isfield(opts,'diagR') || isempty(opts.diagR)
    opts.diagR=false;
end
if ~isfield(opts,'thR') || isempty(opts.thR)
  opts.thR=0;
end
if ~isfield(opts,'outlierReject')
    opts.outlierReject=false;
end
if ~isfield(opts,'indD') || isempty(opts.indD)
  opts.indD=1:nu;%[]; %This means ALL inputs apply to the output equation (D estimate)
  %If given, it should be a logical vector of size 1 x size(U,1), where 1 means include and 0 means exclude
  %Alternatively, it can be a list of included indexes only (e.g. [1,3,4])
end
if ~isfield(opts,'indB') || isempty(opts.indD)
  opts.indB=1:nu; %This means ALL inputs apply to the dynamics equation (B estimate)
  %If given, it should be a logical vector of size 1 x size(U,1), where 1 means include and 0 means exclude
  %Alternatively, it can be a list of included indexes only (e.g. [1,3,4])
end
if ~isfield(opts,'logFlag')
  opts.logFlag=false;
end
end
