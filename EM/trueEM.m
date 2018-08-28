function [A,B,C,D,Q,R,x0,P0,bestLogL]=trueEM(Y,U,Xguess,targetLogL,fastFlag)
%A true EM implementation to do LTI-SSM identification
%INPUT:
%Y is D2 x N
%U is D3 x N
%Xguess - Either the number of states for the system (if scalar) or a guess
%at the initial states of the system (if D1 x N matrix)

if nargin<5 || isempty(fastFlag)
    fastFlag=[];
else
    fastFlag=0; %Disable warnings relating to unstable systems in fast estimation
    w = warning ('off','statKFfast:unstable');
    w = warning ('off','statKSfast:unstable');
end

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
Niter=501;
logl=nan(Niter,1);
logl(1,1)=bestLogL;
if isa(Y,'gpuArray')
    logl=nan(Niter,1,'gpuArray');
end
A=A1; B=B1; C=C1; D=D1; Q=Q1; R=R1; x0=x01; P0=P01;

%Initialize target logL:
if nargin<4 || isempty(targetLogL)
    targetLogL=logl(1);
end


%% ----------------Now, do E-M-----------------------------------------
failCounter=0;
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
        [X1,P1,Pt1,~,~,Xp,Pp,~]=cellfun(@(y,x0,p0,u) statKalmanSmootherFast(y,A1,C1,Q1,R1,x0,p0,B1,D1,u,[],fastFlag),Y,x01,P01,U,'UniformOutput',false);
        if any(cellfun(@(x) any(imag(x(:))~=0),X1))
            error('Complex states') 
        end
    else
        [X1,P1,Pt1,~,~,Xp,Pp,~]=statKalmanSmootherFast(Y,A1,C1,Q1,R1,x01,P01,B1,D1,U,[],fastFlag);
        if any(imag(X1(:))~=0)
            error('Complex states') 
        end
    end
    
    
    %Check improvements:
    l=dataLogLikelihood(Y,U,A1,B1,C1,D1,Q1,R1,Xp,Pp,'approx'); %Passing the Kalman-filtered states and uncertainty makes the computation more efficient
    logl(k+1)=l;
    delta=l-logl(k,1);
    if mod(k,50)==0 %Print info
        pOverTarget=100*(l/targetLogL-1);
        lastChange=l-logl(k-49,1);
    disp(['Iter = ' num2str(k) ', \Delta = ' num2str(lastChange) ', % over target = ' num2str(pOverTarget)])
    end
    improvement=delta>0;
    targetRelImprovement10=(l-logl(max(k-10,1),1))/(targetLogL-l);
    belowTarget=l<targetLogL;
    relImprovementLast10=1-logl(max(k-10,1),1)/l; %Assessing the relative improvement on logl over the last 10 iterations (or less if there aren't as many)
    
    %Check for failure conditions:
    if imag(l)~=0 %This does not happen
        fprintf(['Complex logL, probably ill-conditioned matrices involved. Stopping after ' num2str(k) ' iterations.\n'])
        break
    elseif any(abs(eig(A1))>1)
        %No need to break for unstable systems, usually they converge to a
        %stable system or lack of improvement in logl makes the iteration stop
        %fprintf(['Unstable system detected. Stopping. ' num2str(k) ' iterations.\n'])
        %break
    elseif ~improvement %This should never happen, except that our loglikelihood is approximate, so there can be some error
        if abs(delta)>1e-7 %Do not bother reporting drops within numerical precision
            warning(['logL decreased at iteration ' num2str(k) ', drop = ' num2str(delta)])
        end
        failCounter=failCounter+1;
        %TO DO: figure out why logl sometimes drops a lot on iter 1.
        if failCounter>4
            fprintf(['Dropped 5 times w/o besting the fit. ' num2str(k) ' iterations.\n'])
            break
        end
    else %There was improvement
        if l>=bestLogL
            failCounter=0;
            %If everything went well and these parameters are the best ever: 
            %replace parameters  (notice the algorithm may continue even if 
            %the logl dropped, but in that case we do not save the parameters)
            A=A1; B=B1; C=C1; D=D1; Q=Q1; R=R1; x0=x01; P0=P01; %X=X1; P=P1; %Pt=Pt1;
            bestLogL=l;
        end
    end

    %Check if we should stop early (to avoid wasting time):
    if k>1 && (belowTarget && (targetRelImprovement10)<2e-1) %Breaking if improvement less than 20% of distance to targetLogL, as this probably means we are not getting a solution better than the given target
       fprintf(['unlikely to reach target value. ' num2str(k) ' iterations.\n'])
       break 
    elseif k>1 && (relImprovementLast10)<1e-10 %Considering the system stalled if relative improvement on logl is <1e-7
        fprintf(['increase is within tolerance (local max). '  num2str(k) ' iterations.\n'])
        %disp(['LogL as % of target:' num2str(round(l*100000/targetLogL)/1000)])
        break 
    elseif k==Niter-1
        fprintf(['max number of iterations reached. '  num2str(k) ' iterations.\n'])
    end
    
    %M-step:
    [A1,B1,C1,D1,Q1,R1,x01,P01]=estimateParams(Y,U,X1,P1,Pt1);
end

%%
if fastFlag==0 %Re-enable disabled warnings
    w = warning ('on','statKFfast:unstable');
    w = warning ('on','statKSfast:unstable');
end
end

function maxLperSamplePerDim=fastLogL(predError)
%A fast implementation of the logL when we assume R is set to be optimal 
%over PSD matrices
%Computing in this way is faster, and avoids dealing with numerical issues
%arising from the attempt to compute the optimal R within trueEM.
%See dataLogLikelihood for the exact logl() function and the conditions 
%for this assumption
%INPUT:
%predError: the one-step ahead prediction errors for the model
    [~,N2]=size(predError);
    if sum(predError(:).^2)>1e20 %This can happen when current estimate of system is unstable, and doing fast filtering
        maxLperSamplePerDim=-Inf;
    else
        u=predError/sqrt(N2);
        S=u*u';
        logdetS=mean(log(eig(S)));
        maxLperSamplePerDim = -.5*(1+log(2*pi)+logdetS);
    end
    %maxLperSample = D2*maxLperSamplePerDim;
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
    %[A1,B1,C1,x01,~,Q1,P01] = canonizev3(A1,B1,C1,x01,Q1,P01); 
    %Compute logL:
    logL=dataLogLikelihood(Y,U,A1,B1,C1,D1,Q1,R1,x01,P01,'approx');
end

function [P,Pt]=initCov(X)
    [~,N]=size(X);
    %Initialize covariance to plausible values:
    dX=diff(X');
    Px=cov(dX); 
    P=repmat(Px,1,1,N);
    Pt=repmat(Px',1,1,N);
end

function X=initGuess(Y,U,D1)
if isa(Y,'cell')
    X=cellfun(@(y,u) initGuess(y,u,D1),Y,U,'UniformOutput',false);
else
    D=Y/U;
    if isa(Y,'gpuArray')
        [pp,~,~]=pca(gather(Y-D*U),'Centered','off'); %Can this be done in the gpu?
    else
       [pp,~,~]=pca((Y-D*U),'Centered','off'); %Can this be done in the gpu? 
    end
    X=pp(:,1:D1)';
end
end