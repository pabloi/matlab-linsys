function A=estimateTransitionMatrixv2(X,ord)
%Given a matrix where each column is considered a multi-variate sample of a timeseries,
%estimate the transition matrix such that x_{k+1}= Ax_k +v_k

if nargin<2
  ord=15; %Needs to be odd for a solution to exist always
end
if numel(ord)~=1
    %Assuming ord is the weight vector
    ord=numel(weights);
end

%Option 2: find A to simultaneously fit estimates of all powers of A, much slower, but more accurate
s=size(X,1);
for k=1:ord
  An((k-1)*s+[1:s],:)=X(:,(k+1):end)/X(:,1:end-k);
end
[A]=fitMatrixPowers(An);
end
