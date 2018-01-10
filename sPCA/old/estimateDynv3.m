function [J,Xh,V,K] = estimateDynv3(X, realPolesOnly, nullK, J0)
%estimateDyn for a given vector X, it estimates matrices J,B,V such that
%Xh(:,i+1)=J*Xh(:,i); Xh(:,1)=1; and X~V*Xh + K where J is Jordan Canonical Form
%INPUTS:
%X: D-dimensional time-series [NxD matrix] to be approximated with linear dynamics.
%realPolesOnly: boolean flag indicating if only real poles are to be considered (exponentially decaying terms)
%nullK: boolean flag indicating whether a constant term is to be included as factor.
%J0: can be a scalar which indicates the dimension of J (square) or can be an initial guess of J [has to be square matrix].
%OUTPUTS:
%
%
%Changes in v3: input argument J0 is now mandatory and indicates order
%of dynamics wanted, which no longer needs to be the same as D.
%See also: sPCAv5
% Pablo A. Iturralde - Univ. of Pittsburgh - Last rev: Jun 27th 2017

NN=size(X,2);
if numel(J0)==1 && J0>=1
    order=J0;
    t0=[.1*NN*(1./[1:order]')]; %Initializing to reasonable values, works for realPolesOnly=true
    reps=1;
else
    order=size(J0,1);
    t0=-1./log(eig(J0));
    reps=1;
end

%%
if realPolesOnly % Optimize to find best decaying exponential fits:
    %Bounds & options:
    lb=[zeros(size(t0))];
    ub=[(3*NN*ones(size(t0)))];
    opts=optimoptions('lsqnonlin','FunctionTolerance',1e-18,'OptimalityTolerance',1e-15,'StepTolerance',1e-15,'MaxFunctionEvaluations',1e5,'MaxIterations',3e3,'Display','off');

    %Optimize:
    [xx,resnorm,~,exitflag]=lsqnonlin(@(x) X*projector(x,NN,nullK),t0,lb,ub,opts);
    bestXX=xx;
    bestRes=resnorm;
    %If many repetitions (to try different initial conditions):
    for i=2:reps
        t0=NN*rand(size(t0)); %Uniform distribution
        [xx,resnorm,~,exitflag]=lsqnonlin(@(x) X*projector(x,NN,nullK),t0,lb,ub,opts);
        if resnorm<bestRes
            bestXX=xx;
        end
    end

    %Decompose solution:
    tau=bestXX;
    [Xh]=decays(tau,NN,nullK); %Estimate of states
    J=diag(exp(-1./tau));

    %Find linear regression:
    if nargout>2
        VK=X/Xh;
        V=VK(:,1:order);
        if ~nullK
            K=VK(:,end);
        else
            K=zeros(size(V,1),0); %Empty matrix but [V K] is well defined
        end
    end
else %Allowing for complex & double real poles:
    error('Unimplemented')
end

end

function E=decays(tau,NN,nullK)
    E=exp(-[0:NN-1]./tau);
    if ~nullK
        E=[E;ones(1,NN)];
    end
end

function [P]=projector(tau,NN,nullK) % tau has to be order x 1 vector
    E=decays(tau,NN,nullK);
    EEt=compEEt(E(1:end-(nullK==0),2),NN,nullK);
    P=eye(NN)-(E'/EEt)*E; %Is there a way to avoid directly using E in this computation?
end

function M=compEEt(eTau,NN,nullK)
    alpha=1e-3; %Regularization term: avoids solutions with double poles, which are badly conditioned numerically. 1e-2 keeps the poles ~30 apart, 1e-4 ~4 apart.
    %EEt=(E*E'+alpha*eye(size(E,1)));
    aN=eTau.^NN;
    M=(1-aN*aN')./(1-eTau*eTau') +alpha*eye(size(aN));
    if ~nullK
        E1=(1-eTau.^NN)./(1-eTau);
        M=[M,E1; E1', NN];
    end
end
