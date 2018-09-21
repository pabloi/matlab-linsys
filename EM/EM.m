function [A,B,C,D,Q,R,X,P,bestLogL]=EM(Y,U,Xguess,opts)
%A true EM implementation to do LTI-SSM identification
%INPUT:
%Y is D2 x N
%U is D3 x N
%Xguess - Either the number of states for the system (if scalar) or a guess
%at the initial states of the system (if D1 x N matrix)

if nargin<4
    opts=[];
end
[Niter,robustFlag,fastFlag,convergenceTol,targetTol,targetLogL] = processEMopts(opts);
warning ('off','statKFfast:unstable');
warning ('off','statKFfast:NaNsamples');
warning ('off','statKSfast:unstable');

%% ------------Init stuff:-------------------------------------------
%Define init guess of state:
if isempty(Xguess)
    error('Xguess has to be a guess of the states (D x N matrix) or a scalar indicating the number of states to be estimated')
elseif numel(Xguess)==1 %Xguess is just dimension
    D1=Xguess;
    Xguess=initGuess(Y,U,D1);
end
X=Xguess;

% Init params:
[A1,B1,C1,D1,Q1,R1,x01,P01,bestLogL]=initParams(Y,U,X);

%Initialize log-likelihood register & current best solution:
logl=nan(Niter,1);
logl(1,1)=bestLogL;
if isa(Y,'gpuArray')
    logl=nan(Niter,1,'gpuArray');
end
A=A1; B=B1; C=C1; D=D1; Q=Q1; R=R1; x0=x01; P0=P01; P=repmat(P0,1,1,size(X,2));

%Initialize target logL:
if isempty(targetLogL)
    targetLogL=logl(1);
end


%% ----------------Now, do E-M-----------------------------------------
failCounter=0;
breakFlag=false;
%fh=figure;
disp(['Iter = 1, target logL = ' num2str(targetLogL)])
for k=1:Niter-1
	%E-step: compute the expectation of latent variables given current parameter estimates
    %Note this is an approximation of true E-step in E-M algorithm. The
    %E-step requires to compute the expectation of the likelihood of the data under the
    %latent variables = E(L(Y,X|params)), to then maximize it
    %whereas here we are computing E(X|params) to then maximize L(Y,E(X)|params)
    %logl(k,2)=dataLogLikelihood(Y,U,A,B,C,D,Q,R,X);
	%M-step: find parameters A,B,C,D,Q,R that maximize likelihood of data
    
    %E-step:
    if isa(Y,'cell') %Data is many realizations of same system
        [X1,P1,Pt1,~,~,Xp,Pp,rejSamples]=cellfun(@(y,x0,p0,u) statKalmanSmoother(y,A1,C1,Q1,R1,x0,p0,B1,D1,u,[],fastFlag),Y,x01,P01,U,'UniformOutput',false);
        if any(cellfun(@(x) any(imag(x(:))~=0),X1))
            error('Complex states') 
        end
    else
        [X1,P1,Pt1,~,~,Xp,Pp,rejSamples]=statKalmanSmoother(Y,A1,C1,Q1,R1,x01,P01,B1,D1,U,[],fastFlag);
        if any(imag(X1(:))~=0)
            error('Complex states') 
        end
    end
    
    
    %Check improvements:
    Y2=Y;
    Y2(:,rejSamples)=NaN;
    %sum(rejSamples)
    l=dataLogLikelihood(Y2,U,A1,B1,C1,D1,Q1,R1,Xp,Pp,'approx'); %Passing the Kalman-filtered states and uncertainty makes the computation more efficient
    logl(k+1)=l;
    delta=l-logl(k,1);
    improvement=delta>=0;
    targetRelImprovement50=(l-logl(max(k-50,1),1))/(targetLogL-l);
    belowTarget=max(l,bestLogL)<targetLogL;
    relImprovementLast50=1-logl(max(k-50,1),1)/l; %Assessing the relative improvement on logl over the last 10 iterations (or less if there aren't as many)
    
    %Check for failure conditions:
    if imag(l)~=0 %This does not happen
        msg='Complex logL, probably ill-conditioned matrices involved. Stopping.';
        %fprintf(['Complex logL, probably ill-conditioned matrices involved. Stopping after ' num2str(k) ' iterations.\n'])
        breakFlag=true;
    elseif any(abs(eig(A1))>1)
        %No need to break for unstable systems, usually they converge to a
        %stable system or lack of improvement in logl makes the iteration stop
        %fprintf(['Unstable system detected. Stopping. ' num2str(k) ' iterations.\n'])
        warning('EM:unstableSys','Unstable system detected');
        %break
    elseif ~improvement %This should never happen, except that our loglikelihood is approximate, so there can be some error
        if abs(delta)>1e-6 %Drops of about 1e-6 can be expected because we are
          %computing an approximate logl and because of numerical precision. Report
          %only if drops are larger than this. This value probably is sample-size dependent, so may need adjusting.
          % warning('EM:logLdrop',['logL decreased at iteration ' num2str(k) ', drop = ' num2str(delta)])
        end
        failCounter=failCounter+1;
        %TO DO: figure out why logl sometimes drops a lot on iter 1.
        if failCounter>9
            %fprintf(['Dropped 10 times w/o besting the fit. ' num2str(k) ' iterations.\n'])
            %breakFlag=true;
        end
    else %There was improvement
        if l>=bestLogL
            failCounter=0;
            %If everything went well and these parameters are the best ever: 
            %replace parameters  (notice the algorithm may continue even if 
            %the logl dropped, but in that case we do not save the parameters)
            A=A1; B=B1; C=C1; D=D1; Q=Q1; R=R1; x0=x01; P0=P01; X=X1; P=P1; Pt=Pt1;
            bestLogL=l;
        end
    end

    %Check if we should stop early (to avoid wasting time):
    if k>50 && (belowTarget && (targetRelImprovement50)<targetTol) %Breaking if improvement less than tol of distance to targetLogL
       msg='Unlikely to reach target value. Stopping.';
        %fprintf([ num2str(k) ' iterations.\n'])
       breakFlag=true; 
    elseif k>50 && (relImprovementLast50)<convergenceTol %Considering the system stalled if relative improvement on logl is <tol
        msg='Increase is within tolerance (local max). Stopping.';
        %fprintf(['increase is within tolerance (local max). '  num2str(k) ' iterations.\n'])
        %disp(['LogL as % of target:' num2str(round(l*100000/targetLogL)/1000)])
        breakFlag=true;
    elseif k==Niter-1
        %fprintf(['max number of iterations reached. '  num2str(k) ' iterations.\n'])
        msg='Max number of iterations reached. Stopping.';
        breakFlag=true;
    end
    
    %Print some info
    step=50;
    if mod(k,step)==0 || breakFlag %Print info
        pOverTarget=100*(l/targetLogL-1);
        if k>=step && ~breakFlag
            lastChange=l-logl(k+1-step,1);
            disp(['Iter = ' num2str(k) ', \Delta logL = ' num2str(lastChange) ', % over target = ' num2str(pOverTarget)])
        else %k==1 || breakFlag
            l=bestLogL;
            pOverTarget=100*(l/targetLogL-1);
            disp(['Iter = ' num2str(k) ', logL = ' num2str(l) ', % over target = ' num2str(pOverTarget)])
            if breakFlag
            fprintf([msg ' \n'])
            end
        end
    end
    if breakFlag
        break
    end
    %M-step:
    [A1,B1,C1,D1,Q1,R1,x01,P01]=estimateParams(Y,U,X1,P1,Pt1,robustFlag);
    %[A1,B1,C1,x01,~,Q1,P01] = canonizev2(A1,B1,C1,x01,Q1,P01);
end

%%
if fastFlag==0 %Re-enable disabled warnings
    warning ('on','statKFfast:unstable');
    warning ('on','statKFfast:NaNsamples');
    warning ('on','statKSfast:unstable');
end
end

function [A1,B1,C1,D1,Q1,R1,x01,P01,logL]=initParams(Y,U,X)

if isa(Y,'cell')
    [P,Pt]=cellfun(@initCov,X,'UniformOutput',false);
else
    %Initialize covariance to plausible values:
    [P,Pt]=initCov(X);

    %Move things to gpu if needed
    if isa(Y,'gpuArray')
        U=gpuArray(U);
        X=gpuArray(X);
        P=gpuArray(P);
        Pt=gpuArray(Pt);
    end
end

    %Initialize guesses of A,B,C,D,Q,R
    [A1,B1,C1,D1,Q1,R1,x01,P01]=estimateParams(Y,U,X,P,Pt);
    %Make sure scaling is appropriate:
    [A1,B1,C1,x01,~,Q1,P01] = canonizev2(A1,B1,C1,x01,Q1,P01); 
    %Compute logL:
    logL=dataLogLikelihood(Y,U,A1,B1,C1,D1,Q1,R1,x01(:,1),P01(:,:,1),'approx');
end

function [P,Pt]=initCov(X)
    [~,N]=size(X);
    %Initialize covariance to plausible values:
    dX=diff(X');
    Px=(dX'*dX)/N; 
    P=repmat(Px,1,1,N);
    %Px1=(dX(2:end,:)'*dX(1:end-1,:));
    Pt=repmat(.2*diag(diag(Px)),1,1,N);
end

function X=initGuess(Y,U,D1)
if isa(Y,'cell')
    X=cellfun(@(y,u) initGuess(y,u,D1),Y,U,'UniformOutput',false);
else
    idx=~any(isnan(Y));
    D=Y(:,idx)/U(:,idx);
    if isa(Y,'gpuArray')
        [pp,~,~]=pca(gather(Y(:,idx)-D*U(:,idx)),'Centered','off'); %Can this be done in the gpu?
    else
       [pp,~,~]=pca((Y(:,idx)-D*U(:,idx)),'Centered','off'); %Can this be done in the gpu? 
    end
    X=nan(D1,size(Y,2));
    X(:,idx)=pp(:,1:D1)';
    X(:,~idx)=interp1(find(idx),pp(:,1:D1),find(~idx))';
end
end
