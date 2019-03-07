function [A,B,C,D,Q,R,X,P,bestLogL,outLog]=EM(Y,U,Xguess,opts,Pguess)
%A true EM implementation to do LTI-SSM identification
%INPUT:
%Y is D2 x N
%U is D3 x N
%Xguess - Either the number of states for the system (if scalar) or a guess
%at the initial states of the system (if D1 x N matrix)

%Scaling: (Note: scaling is not important to EM, the optimal solution is
%scale invariant. However, some constants here have been fine-tuned
%presuming a certain scaling of the data for convergence issues.)
if ~isa(Y,'cell')
    scale=sqrt(nanmean(Y.^2,2));
    Y=Y./scale;
else
    scale=sqrt(nanmean(cell2mat(Y).^2,2));
    Y=cellfun(@(x) x./scale,Y,'UniformOutput',false);
end

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
outLog=[];

%Process opts:
if isa(U,'cell')
  Nu=size(U{1},1);
else
  Nu=size(U,1);
end
[opts] = processEMopts(opts,Nu); %This is a fail-safe to check for proper options being defined.
if opts.fastFlag~=0 && ( (~isa(Y,'cell') && any(isnan(Y(:)))) || (isa(Y,'cell') && any(any(isnan(cell2mat(Y)))) ) )
   warning('EM:fastAndLoose','Requested fast filtering but data contains NaNs. No steady-state can be found for filtering. Filtering will not be exact. Proceed at your own risk.')
  %opts.fastFlag=0; %No fast-filtering in nan-filled data
elseif opts.fastFlag~=0 && opts.fastFlag~=1
    warning('EM:fastFewSamples','Requested an exact number of samples for fast filtering. This is guaranteed to be equivalent to fast filtering only if the slowest time-constant of the system is much smaller than the requested number of samples, otherwise this is an appoximation.')
end
if opts.logFlag
  %diary(num2str(round(now*1e5)))
  outLog.opts=opts;
  tic
end

%Disable some annoying warnings related to fast filtering (otherwise these 
%warnings appear on each iteration when the Kalman filter is run):
warning('off','statKFfast:unstable');
warning('off','statKFfast:NaN');
warning('off','statKSfast:unstable');
warning('off','statKSfast:fewSamples');

%% ------------Init stuff:-------------------------------------------
% Init params:
 [A1,B1,C1,D1,Q1,R1,x01,P01]=initEM(Y,U,Xguess,opts,Pguess);
 [X1,P1,Pt1,~,~,~,~,~,bestLogL]=statKalmanSmoother(Y,A1,C1,Q1,R1,x01,P01,B1,D1,U,opts);

%Initialize log-likelihood register & current best solution:
logl=nan(opts.Niter,1);
logl(1)=bestLogL;
if isa(Y,'gpuArray')
    logl=nan(Niter,1,'gpuArray');
end
A=A1; B=B1; C=C1; D=D1; Q=Q1; R=R1; x0=x01; P0=P01; P=P1; Pt=Pt1; X=X1;

%Initialize target logL:
if isempty(opts.targetLogL)
    opts.targetLogL=logl(1);
end
%figure;
%hold on;
%% ----------------Now, do E-M-----------------------------------------
breakFlag=false;
improvement=true;
disp(['Iter = 1, target logL = ' num2str(opts.targetLogL,8) ', current logL=' num2str(bestLogL,8) ', \tau =' num2str(-1./log(sort(eig(A)))')])
for k=1:opts.Niter-1
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
    if isa(Y,'cell') %Data is many realizations of same system
        [X1,P1,Pt1,~,~,~,~,~,l1]=cellfun(@(y,x0,p0,u) statKalmanSmoother(y,A1,C1,Q1,R1,x0,p0,B1,D1,u,opts),Y,x01,P01,U,'UniformOutput',false);
        if any(cellfun(@(x) any(imag(x(:))~=0),X1))
          msg='Complex states detected, stopping.';
          breakFlag=true;
        elseif any(cellfun(@(x) isnan(sum(x(:))),X1))
          msg='States are NaN, stopping.';
          breakFlag=true;
        end
        sampleSize=cellfun(@(y) size(y,2),Y);
        l=(cell2mat(l1)*sampleSize')/sum(sampleSize);
    else
        [X1,P1,Pt1,~,~,~,~,~,l]=statKalmanSmoother(Y,A1,C1,Q1,R1,x01,P01,B1,D1,U,opts);
        if any(imag(X1(:))~=0)
            msg='Complex states detected, stopping.';
            breakFlag=true;
        elseif isnan(sum(X1(:)))
            msg='States are NaN, stopping.';
            breakFlag=true;
        end
    end


    %Check improvements:
    logl(k+1)=l;
    delta=l-logl(k);
    improvement=delta>=0;
    targetRelImprovement50=(l-logl(max(k-50,1),1))/(opts.targetLogL-logl(max(k-50,1),1));
    belowTarget=max(l,bestLogL)<opts.targetLogL;
    relImprovementLast50=l-logl(max(k-50,1)); %Assessing the improvement on logl over the last 50 iterations (or less if there aren't as many)

    %Check for warning conditions:
    if ~improvement %This should never happen, except that our loglikelihood is approximate, so there can be some error
        %Update Jan 2019: I think with the latest versions the logL is now
        %exact.
        if abs(delta)>1e-5 %Drops of about 1e-5 can be expected because (conjectures):
          %1) we are computing an approximate logl (which differs from the exact one, especially at the early stages of EM)
          %2) logL is only guaranteed to increase if there is no structural model mismatch (e.g. data having non-gaussian observation noise). Although it may work in other circumstances. Need to prove.
          %3)numerical precision.
          %Report only if drops are larger than this. This value probably is sample-size dependent, so may need adjusting.
          warning('EM:logLdrop',['logL decreased at iteration ' num2str(k) ', drop = ' num2str(delta)])
        end
    end

    %Check for failure conditions:
    if imag(l)~=0 %This does not happen
        msg='Complex logL, probably ill-conditioned matrices involved. Stopping.';
        breakFlag=true;
    elseif l>=bestLogL %There was improvement
        %If everything went well and these parameters are the best ever:
        %replace parameters  (notice the algorithm may continue even if
        %the logl dropped, but in that case we do not save the parameters)
        A=A1; B=B1; C=C1; D=D1; Q=Q1; R=R1; x0=x01; P0=P01; X=X1; P=P1; Pt=Pt1;
        bestLogL=l;
    end

    %Check if we should stop early (to avoid wasting time):
    if k>50 && (belowTarget && (targetRelImprovement50)<opts.targetTol) && ~opts.robustFlag%Breaking if improvement less than tol of distance to targetLogL
       msg='Unlikely to reach target value. Stopping.';
       breakFlag=true;
    elseif k>50 && (relImprovementLast50)<opts.convergenceTol && ~opts.robustFlag %Considering the system stalled if relative improvement on logl is <tol
        msg='Increase is within tolerance (local max). Stopping.';
        breakFlag=true;
    elseif k==opts.Niter-1
        msg='Max number of iterations reached. Stopping.';
        breakFlag=true;
    end

    %Print some info
    step=100;
    if mod(k,step)==0 || breakFlag %Print info
        pOverTarget=100*((l-opts.targetLogL)/abs(opts.targetLogL));
        if k>=step && ~breakFlag
            lastChange=l-logl(k+1-step,1);
            disp(['Iter = ' num2str(k)  ', logL = ' num2str(l,8) ', \Delta logL = ' num2str(lastChange,3) ', % over target = ' num2str(pOverTarget,3) ', \tau =' num2str(-1./log(sort(eig(A1)))',3)])
            %sum(rejSamples)
        else %k==1 || breakFlag
            l=bestLogL;
            pOverTarget=100*((l-opts.targetLogL)/abs(opts.targetLogL));
            disp(['Iter = ' num2str(k) ', logL = ' num2str(l,8) ', % over target = ' num2str(pOverTarget) ', \tau =' num2str(-1./log(sort(eig(A)))')])
            if breakFlag; fprintf([msg ' \n']); end
        end
    end
    if breakFlag && ~opts.robustFlag
        break
    end
    %M-step:
    [A1,B1,C1,D1,Q1,R1,x01,P01]=estimateParams(Y,U,X1,P1,Pt1,opts);
end %for loop

%%
if opts.fastFlag==0 %Re-enable disabled warnings
    warning ('on','statKFfast:unstable');
    warning ('on','statKFfast:NaNsamples');
    warning ('on','statKSfast:unstable');
end
if opts.logFlag
  outLog.vaps(k,:)=sort(eig(A1));
  outLog.runTime(k)=toc;
  outLog.breakFlag=breakFlag;
  outLog.msg=msg;
  outLog.bestLogL=bestLogL;
  %diary('off')
end

%% Restore scale:
C=C.*scale;
D=D.*scale;
R=scale.*R.*scale';
end  %Function
