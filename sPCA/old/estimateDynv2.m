function [J,B,Xh] = estimateDynv2(X, realPolesOnly, nullB, J0)
%estimateDyn for a given vector X, it estimates matrices J,B,V such that
%Xh(:,i+1)=J*Xh(:,i)+B; Xh(:,1)=1; and X~V*Xh where J is Jordan Canonical Form
%v2 is a lot more efficient

NN=size(X,2);
if nargin<4 || isempty(J0)
    order=size(X,1); 
    if ~nullB
        order=order-1;
    end
else
    order=size(J0,1); %Expected square matrix
end


%%
if realPolesOnly % Optimize to find best decaying exponential fits:
    %Init:
    if nargin<4 || isempty(J0)
        t0=[.1*NN*(1./[1:order]')]; %Initializing to reasonable values
        reps=10;
    else
        t0=-1./log(eig(J0));
        reps=1;
    end

    %Bounds & options:
    lb=[zeros(size(t0))];
    ub=[(3*NN*ones(size(t0)))];
    opts=optimoptions('lsqnonlin','FunctionTolerance',1e-18,'OptimalityTolerance',1e-15,'StepTolerance',1e-15,'MaxFunctionEvaluations',1e5,'MaxIterations',3e3,'Display','off');

    %Optimize:
    [xx,resnorm,~,exitflag]=lsqnonlin(@(x) X*projector(x,NN,nullB),t0,lb,ub,opts);
    bestXX=xx;
    bestRes=resnorm;
    %If many repetitions (to try different initial conditions):
    for i=2:reps
        t0=NN*rand(size(t0)); %Uniform distribution
        [xx,resnorm,~,exitflag]=lsqnonlin(@(x) X*projector(x,NN,nullB),t0,lb,ub,opts);
        if resnorm<bestRes
            bestXX=xx;
        end
    end

    %Decompose solution:
    tau=bestXX;
    [Xh]=decays(tau,NN,nullB); %Estimate of states
    J=diag(exp(-1./tau));
    
    %By definition, this solution is the one that satisfies B=0 (Xinf=0)
    B=zeros(order,1);

else
    error('Unimplemented')
end

end

function E=decays(tau,NN,nullB)
    E=exp(-[0:NN-1]./tau); 
    if ~nullB
        E=[E;ones(1,NN)];
    end
end

function [P]=projector(tau,NN,nullB) % tau has to be order x 1 vector
    E=decays(tau,NN,nullB);
    EEt=compEEt(E(1:end-(nullB==0),2),NN,nullB);
    P=eye(NN)-(E'/EEt)*E; %Is there a way to avoid directly using E in this computation?
end

function M=compEEt(eTau,NN,nullB)
    alpha=1e-3; %Regularization term: avoids solutions with double poles, which are badly conditioned numerically. 1e-2 keeps the poles ~30 apart, 1e-4 ~4 apart.
    %EEt=(E*E'+alpha*eye(size(E,1)));
    aN=eTau.^NN;
    M=(1-aN*aN')./(1-eTau*eTau') +alpha*eye(size(aN));
    if ~nullB
        E1=(1-eTau.^NN)./(1-eTau);
        M=[M,E1; E1', NN];
    end
end