function [A,B,C,D,Q,R,X,P]=trueEM(Y,U,Xguess,targetLogL)
%A true EM implementation to do LTI-SSM identification
%INPUT:
%Y is D2 x N
%U is D3 x N
%Xguess - Either the number of states for the system (if scalar) or a guess
%at the initial states of the system (if D1 x N matrix)


[D2,N]=size(Y);
%Initialize guesses of A,B,C,D,Q,R
D=Y/U;
X=Xguess;
if numel(X)==1
    D1=X;
    [pp,~,~]=pca(Y-D*U,'Centered','off');
    X=pp(:,1:D1)';
else
    D1=size(X,1);
end
%Starting point:
[A,B,C,D,Q,R,x0,P0]=estimateParams(Y,U,X,zeros(D1,D1,N),zeros(D1,D1,N-1));


debug=false;
logl=nan(501,1);
logl(1,1)=dataLogLikelihood(Y,U,A,B,C,D,Q,R,x0,P0);
if nargin<4 || isempty(targetLogL)
    targetLogL=logl(1);
end

%Now, do E-M
for k=1:size(logl,1)-1
	%E-step: compute the expectation of latent variables given current parameter estimates
    %Note this is an approximation of true E-step in E-M algorithm. The
    %E-step requires to compute the expectation of the likelihood of the data under the
    %latent variables = E(L(Y,X|params)), to then maximize it
    %whereas here we are computing E(X|params) to then maximize L(Y,E(X)|params)
    %logl(k,2)=dataLogLikelihood(Y,U,A,B,C,D,Q,R,X);
	%M-step: find parameters A,B,C,D,Q,R that maximize likelihood of data
    
    %E-step:
    [X1,P1,Pt1,~,~,Xp,Pp,~]=statKalmanSmoother(Y,A,C,Q,R,x0,P0,B,D,U);
    if any(imag(X1(:)))~=0
       warning('Complex states') 
    end
    %M-step:
    [A1,B1,C1,D1,Q1,R1,x01,P01]=estimateParams(Y,U,X1,P1,Pt1);
    if debug
        disp(['LogL as % of target:' num2str(round(l*100000/targetLogL)/1000)])
        l=dataLogLikelihood(Y,U,A1,B1,C1,D1,Q1,R1,Xp,Pp);
        if l<logl(k+1,1) %Make only partial updates that do increase likelihood, avoids some numerical issues
            warning('logL did not increase. Doing partial updates.')
            ch=false;
           if dataLogLikelihood(Y,U,A,B,C1,D1,Q,R1,x0,P0)>logl(k+1,1)
               disp('Updating C,D,R')
               C=C1; D=D1; R=R1;
               ch=true;
           end
           if dataLogLikelihood(Y,U,A1,B1,C,D,Q,R,x0,P0)>logl(k+1,1)
               disp('Updating A,B')
               A=A1; B=B1;
               ch=true;
           end
           if dataLogLikelihood(Y,U,A,B,C,D,Q1,R,x0,P0)>logl(k+1,1)
               disp('Updating Q')
               Q=Q1;
               ch=true;
           end
           if dataLogLikelihood(Y,U,A,B,C,D,Q,R,x01,P01)>logl(k+1,1)
               disp('Updating x0,P0')
               x0=x01; P0=P01;
               ch=true;
           end
           if ~ch
               warning('logL did not increase. Stopping')
               break
           end
        %else %Update all
        %   A=A1; B=B1; C=C1; D=D1; Q=Q1; R=R1; x0=x01; P0=P01; 
        end
        %l=dataLogLikelihood(Y,U,A,B,C,D,Q,R,x0,P0)
    %else
    %    [A1,B1,C1,D1,Q1,R1,x01,P01]=estimateParams(Y,U,X1,P1,Pt1);
    end
    
    %Check improvements:
    l=dataLogLikelihood(Y,U,A1,B1,C1,D1,Q1,R1,Xp,Pp); %Passing the Kalman-filtered states and uncertainty makes the computation more efficient
    logl(k+1)=l;
    improvement=l>=logl(k,1);
    targetRelImprovement=(l-logl(k,1))/(targetLogL-l);
    belowTarget=l<targetLogL;
    relImprovementLast10=1-logl(max(k-10,1),1)/l; %Assessing the relative improvement on logl over the last 10 iterations (or less if there aren't as many)
    %Check for failure conditions:
    if imag(l)~=0
        warning('Complex logL, probably ill-conditioned matrices involved. Stopping.')
        break
    elseif ~improvement %This should never happen, except that our loglikelihood is approximate, so there can be some rounding error
        if l<logl(max(k-3,1),1) %If the logl dropped below what it was 3 steps before, then we probably have a real issue (Best case: local max)
            warning('logL decreased. Stopping')
            break
        end
    end
    %If everything went well: replace parameters 
    A=A1; B=B1; C=C1; D=D1; Q=Q1; R=R1; x0=x01; P0=P01; X=X1; P=P1; %Pt=Pt1;
    %Check if we should stop early (to avoid wasting time):
    if k>1 && (belowTarget && (targetRelImprovement)<2e-2) %Breaking if improvement less than 2% of distance to targetLogL, as this probably means we are not getting a solution better than the given target
       warning('logL unlikely to reach target value. Stopping')
       break 
    elseif k>1 && (relImprovementLast10)<1e-7 %Considering the system stalled if relative improvement on logl is <1e-7
        warning('logL increase is within tolerance (local max). Stopping')
        %disp(['LogL as % of target:' num2str(round(l*100000/targetLogL)/1000)])
        break 
    end
end
%figure
%subplot(2,1,1)
%plot(reshape(logl',numel(logl),1))
%subplot(2,1,2)
%plot(logl(:,2)-logl(:,1))
