function [A,B,C,D,Q,R,X,P]=randomStartEM(Y,U,D1,Nreps,method)
if nargin<4 || isempty(Nreps)
    Nreps=20;
end
if nargin<5 || isempty(method)
   method='fast'; 
end %TODO: if method is given, check that it is 'true' or 'fast'
bestLL=-Inf;
N=size(Y,2);

for i=1:Nreps
    Xguess=randn(D1,N);
    switch method
        case 'fast'
            [Ai,Bi,Ci,Di,Qi,Ri,Xi,Pi]=fastEM(Y,U,Xguess);
        case 'true'
            [Ai,Bi,Ci,Di,Qi,Ri,Xi,Pi]=trueEM(Y,U,Xguess);
    end
    logl=dataLogLikelihood(Y,U,Ai,Bi,Ci,Di,Qi,Ri,Xi);
    if logl>bestLL
        A=Ai; B=Bi; C=Ci; D=Di; Q=Qi; R=Ri; X=Xi; P=Pi;
        bestLL=logl;
    end
end

end