function [Niter,robustFlag,fastFlag,convergenceTol,targetTol,targetLogL] = processEMopts(opts)

if ~isfield(opts,'Niter')
    Niter=500; %Max number of iters
else
    Niter=opts.Niter;
end

if ~isfield(opts,'robustFlag')
    robustFlag=false; % Non-robust behavior
else
    robustFlag=opts.robustFlag;
end

if ~isfield(opts,'fastFlag')
    fastFlag=0; %statKF/KS auto-select fast samples
else
    fastFlag=opts.fastFlag;
end

if ~isfield(opts,'convergenceTol')
    convergenceTol=1e-9; % 1e-9 minimum relative improvement in logL every 50 strides
else
    convergenceTol=opts.convergenceTol;
end

if ~isfield(opts,'targetTol')
    targetTol=3e-1; % 30% minimum improvement towards target every 50 iters
else
    targetTol=opts.targetTol;
end

if ~isfield(opts,'targetLogL')
    targetLogL=[]; % 30% minimum improvement towards target every 50 iters
else
    targetLogL=opts.targetLogL;
end
    
end

