function [opts] = processEMopts(opts)

if ~isfield(opts,'Niter')
    opts.Niter=500; %Max number of iters
end

if ~isfield(opts,'robustFlag')
    opts.robustFlag=false; % Non-robust behavior
end

if ~isfield(opts,'fastFlag')
    opts.fastFlag=0; %statKF/KS auto-select fast samples
end

if ~isfield(opts,'convergenceTol')
    opts.convergenceTol=1e-10; % 1e-9 minimum relative improvement in logL every 50 strides
end

if ~isfield(opts,'targetTol')
    opts.targetTol=5e-1; % 50% minimum improvement towards target every 50 iters
end

if ~isfield(opts,'targetLogL')
    opts.targetLogL=[];
end

if ~isfield(opts,'sphericalR')
    opts.sphericalR=false;
end
if ~isfield(opts,'outlierReject')
    opts.outlierReject=false;
end
end
