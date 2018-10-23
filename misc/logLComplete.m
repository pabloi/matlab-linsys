function logLperSamplePerDim=logLcomplete(Y,U,A,B,C,D,Q,R,X,method)
%Evaluates the likelihood of the data under a given model

if nargin<10 || isempty(method)
    method='approx'; %Computing approx version by default, exact is too slow
end

%Check if R is psd, and do cholesky decomp:
[cR,r]=mycholcov(R);
if r~=size(R,1)
    warning('R is not PD, this can end badly')
    %This is ok as long as R +C*P*C' is strictly positive definite.
    %Otherwise, nothing good can happen here, unless magically the residuals z have no projection on the null-space of R+C*P*C', and then we *could* compute a pseudo-likelihood by reducing the output to exclude the projection onto the null space, and reducing R+C*P*C' accordingly. However, that changes the dimension of the problem.
end

if isa(Y,'cell') %Case where input/output data corresponds to many realizations of system, requires Y,U,x0,P0 to be cells of same size
    logLperSamplePerDim=cellfun(@(y,u,x0,p0) dataLogLikelihood(y,u,A,B,C,D,Q,R,x0,p0,method),Y,U,X0,P0);
    sampleSize=cellfun(@(y) size(y,2),Y);
    logLperSamplePerDim=(logLperSamplePerDim*sampleSize')/sum(sampleSize);
else
    %Complete logL: p({y},{x}|params)
    z=Y-(C*X+D*U); %Output residuals
    idx=~any(isnan(Y));
    z=z(:,idx); %Removing NaN samples
    w=X(:,2:end)-(A*X(:,1:end-1)+B*U(1:end-1)); %State residuals

    zcR=cR'\z;
    zscores=sum(zcR.^2,1);
    [D2,N2]=size(z);
    logdetR=2*sum(log(diag(cR)));
    logLz=-.5*sum(zscores +logdetR +D2*log(2*pi));

    cQ=mycholcov(Q);
    wcR=cQ'\w;
    zscores=sum(wcR.^2,1);
    [D1,N2]=size(w);
    logdetQ=2*sum(log(diag(cQ)));
    logLw=-.5*sum(zscores +logdetQ +D1*log(2*pi));

    logLperSamplePerDim=(logLw+logLz)/(N2*D2);

end
end
