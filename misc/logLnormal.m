function [logL,z2]=logLnormal(y,Sigma,cholInvSigma)
  %Computes the log-likelihood of a multivariate, zero-mean, normal distr.
  %For a non-zero mean distribution, just use y=x-mu
  %INPUTS:
  %y: observations (D x N matrix: N samples, D-dimensional obs.)
  %Sigma: D x D covariance matrix of multinormal. Optional if argument 3 is provided
  %cholInvSigma: cholesky decomp of inverse covariance matrix
  %log2Pi=1.83787706640934529;
  halfLog2Pi=0.91893853320467268;
  if nargin<3
    cholInvSigma=pinvchol(Sigma)';
  end
  halfLogdetSigma= sum(log(diag(cholInvSigma)));
  icSy=cholInvSigma*y;
  z2=sum(icSy.^2,1); %z^2 scores
  logL=-.5*(z2) +halfLogdetSigma-size(y,1)*halfLog2Pi; 
end
