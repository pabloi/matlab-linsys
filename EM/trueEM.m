function [A,B,C,D,Q,R,X,P]=trueEM(Y,U,Xguess)
%A true EM implementation to do LTI-SSM identification
%INPUT:
%Y is D2 x N
%U is D3 x N
%Xguess - Either the number of states for the system (if scalar) or a guess
%at the initial states of the system (if D1 x N matrix)


[D2,N]=size(Y);
%Initialize guesses of A,B,C,D,Q,R
D=Y/U;
if numel(Xguess)==1
    D1=Xguess;
    [pp,~,~]=pca(Y-D*U,'Centered','off');
    Xguess=pp(:,1:D1)';
else
    D1=size(Xguess,1);
end
[A,B,C,D,Q,R,x0,P0]=estimateParams(Y,U,Xguess,zeros(D1,D1,N),zeros(D1,D1,N-1));

debug=false;
logl=nan(51,1);
logl(1,1)=dataLogLikelihood(Y,U,A,B,C,D,Q,R,x0,P0);

%Now, do E-M
for k=1:size(logl,1)-1
	%E-step: compute the expectation of latent variables given current parameter estimates
    %Note this is an approximation of true E-step in E-M algorithm. The
    %E-step requires to compute the expectation of the likelihood of the data under the
    %latent variables = E(L(Y,X|params)), to then maximize it
    %whereas here we are computing E(X|params) to then maximize L(Y,E(X)|params)
    %logl(k,2)=dataLogLikelihood(Y,U,A,B,C,D,Q,R,X);
	%M-step: find parameters A,B,C,D,Q,R that maximize likelihood of data
	[X,P,Pt,~,~,Xp,Pp,~]=statKalmanSmoother(Y,A,C,Q,R,x0,P0,B,D,U);
    l=dataLogLikelihood(Y,U,A,B,C,D,Q,R,Xp,Pp); %Passing the Kalman-filtered states and uncertainty makes the computation more efficient
    logl(k+1)=l;
    if l<logl(k,1)
       warning('logL decreased. Stopping')
       break 
    elseif k>1 & (1-logl(k,1)/l)<1e-5
        warning('logL increase is within tolerance (local max). Stopping')
       break 
    end
    if debug
        [A1,B1,C1,D1,Q1,R1,x01,P01]=estimateParams(Y,U,X,P,Pt);
        l=dataLogLikelihood(Y,U,A1,B1,C1,D1,Q1,R1,x01,P01);
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
        else %Update all
           A=A1; B=B1; C=C1; D=D1; Q=Q1; R=R1; x0=x01; P0=P01; 
        end
        l=dataLogLikelihood(Y,U,A,B,C,D,Q,R,x0,P0)
    else
        [A,B,C,D,Q,R,x0,P0]=estimateParams(Y,U,X,P,Pt);
    end
end
%figure
%subplot(2,1,1)
%plot(reshape(logl',numel(logl),1))
%subplot(2,1,2)
%plot(logl(:,2)-logl(:,1))
