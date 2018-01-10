function [J,B,Xh,V] = estimateDyn(X, realPolesOnly, nullB, J0)
%estimateDyn for a given vector X, it estimates matrices J,B,V such that
%Xh(:,i+1)=J*Xh(:,i)+B; Xh(:,1)=1; and X~V*Xh where J is Jordan Canonical Form

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
        t0=[.4*NN*([1:order]'/order).^2];
    else
        t0=-1./log(eig(J0));
    end
    E0=myfun(t0,NN,nullB);
    M0=(X)/E0;
    xx0=[M0(:); t0];

    %Bounds & options:
    lb=[-Inf*ones(size(M0(:))); zeros(size(t0))];
    ub=[Inf*ones(size(M0(:))); (3*NN*ones(size(t0)))];
    opts=optimoptions('lsqnonlin','FunctionTolerance',1e-12,'OptimalityTolerance',1e-15,'StepTolerance',1e-15,'MaxFunctionEvaluations',1e5,'MaxIterations',3e3);

    %Optimize:
    [xx,~,~,exitflag]=lsqnonlin(@(x) X - reshape(x(1:numel(M0)),size(M0,1),size(M0,2))*myfun(x(end-order+1:end),NN,nullB),xx0,lb,ub,opts);

    %Decompose solution:
    tau=xx(end-order+1:end);
    V=reshape(xx(1:numel(M0)),size(M0,1),size(M0,2));
    Xh=myfun(tau,NN,nullB); %Estimate of states

    %Equivalent to: 
    J=diag(exp(-1./tau));
    
    %Scale solution such that Xinf=1 (X(0)=0 is assumed)
    B=(eye(size(J))-J)*ones(order,1);

else
    error('Unimplemented')
end

end

function E=myfun(tau,NN,nullB) %M has to be order x order matrix, tau has to be order x 1 vector
    E=exp(-bsxfun(@rdivide,[0:NN-1],tau));
    if ~nullB
        E=1-E;
        E=[E;ones(1,NN)];
    end
end