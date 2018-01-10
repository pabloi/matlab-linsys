function [J,Xh,V,K] = estimateDynv4(X, realPolesOnly, U, J0)
%estimateDyn for a given vector X, it estimates matrices J,B,V such that
%Xh(:,i+1)=J*Xh(:,i) + K*U; Xh(:,1)=1; and X~V*Xh where J is Jordan Canonical Form
%INPUTS:
%X: D-dimensional time-series [NxD matrix] to be approximated with linear dynamics.
%realPolesOnly: boolean flag indicating if only real poles are to be considered (exponentially decaying terms)
%U: input matrix/vector of length N. If empty we assume U=0;
%J0: can be a scalar which indicates the dimension of J (square) or can be an initial guess of J [has to be square matrix].
%OUTPUTS:
%
%
%Changes in v3: input argument J0 is now mandatory and indicates order
%of dynamics wanted, which no longer needs to be the same as D.
%Changes in v4: input 'nullK' is now called U, and can be a matrix of
%length equal to length of X. Fitted model now supports arbitrary U, when
%previously it was only U=1 or U=0
%See also: sPCAv6
% Pablo A. Iturralde - Univ. of Pittsburgh - Last rev: Aug 22nd 2017

NN=size(X,2);
if numel(J0)==1
    order=J0;
    t0=[.1*NN*(1./[1:order]')]; %Initializing to reasonable values, works for realPolesOnly=true
    reps=10;
else
    order=size(J0,1);
    t0=-1./log(eig(J0));
    reps=1;
end

if (~isempty(U) && all(U(:)==0)) || isempty(U)
    U=zeros(0,NN);
    %In this case use v3 which is more efficient:
    [J,Xh,V,K] = estimateDynv3(X, realPolesOnly, true, J0);
    return
elseif all(reshape((U-U(:,1)),numel(U),1)==0)
    %In this case use v3 which is more efficient:
    [J,Xh,V,K] = estimateDynv3(X, realPolesOnly, false, J0);
    return
end

error('This function is not yet implemented')
%Notes to self:
%In order to make this work, I need to change the assumption that states
%evolve as exponentially decaying functions (true if U=constant) and
%project onto that.
%Instead of using decays_old() would need to compute state evolution
%step-by=step for the arbitrary input. If the input is scalar this is easy,
%as K can be assumed to be a vector of 1 (or can it?). If not, we need to
%estimate K concurrently with tau (previously only tau needed optimization)
%If input is piece-wise constant, we could make it efficient by calling on
%decays_old() for each piece.

%%
if realPolesOnly % Optimize to find best decaying exponential fits:
    %Bounds & options:
    lb=[zeros(size(t0))];
    ub=[(3*NN*ones(size(t0)))];
    opts=optimoptions('lsqnonlin','FunctionTolerance',1e-18,'OptimalityTolerance',1e-15,'StepTolerance',1e-15,'MaxFunctionEvaluations',1e5,'MaxIterations',3e3,'Display','off');

    %Optimize:
    [xx,resnorm,~,exitflag]=lsqnonlin(@(x) X*projector(x,NN,U),t0,lb,ub,opts);
    bestXX=xx;
    bestRes=resnorm;
    %If many repetitions (to try different initial conditions):
    for i=2:reps
        t0=NN*rand(size(t0)); %Uniform distribution
        [xx,resnorm,~,exitflag]=lsqnonlin(@(x) X*projector(x,NN,U),t0,lb,ub,opts);
        if resnorm<bestRes
            bestXX=xx;
        end
    end

    %Decompose solution:
    tau=bestXX;
    [Xh]=decays(tau,NN,U); %Estimate of states
    J=diag(exp(-1./tau));
    
    %Find linear regression:
    if nargout>2
        VK=X/Xh;
        V=VK(:,1:order);
        K=VK(:,end-size(U,1):end); %May be an empty array
    end
else %Allowing for complex & double real poles:
    error('Unimplemented')
end

end

function E=decays(tau,NN,U)
    if size(U,1)==1 %This is painfully slow
        E=ones(length(tau),NN);
        K=ones(length(tau),1);
        eTau=exp(-1./tau);
        for i=2:NN
            E(:,i)=eTau.*E(:,i-1) +K*U(i-1);
        end
    else
        error('Unimplemented')
    end
end

function [P]=projector(tau,NN,U) % tau has to be order x 1 vector
    E=decays(tau,NN,U);
    EEt=E*E'; %No way around this inefficient computation in the general case
    P=eye(NN)-(E'/EEt)*E; %Is there a way to avoid directly using E in this computation?
end

% function E=decays_old(tau,NN,U)
%     E=exp(-[0:NN-1]./tau); 
%     E=[E;U]; %Is this faster in the case isempty(U)==true?
% %     if ~isempty(U)
% %         E=[E;U];
% %     end
% end