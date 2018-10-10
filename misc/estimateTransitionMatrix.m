function A=estimateTransitionMatrix(X,ord)
%Given a matrix where each column is considered a multi-variate sample of a timeseries,
%estimate the transition matrix such that x_{k+1}= Ax_k +v_k

if nargin<2
  ord=15; %Needs to be odd for a solution to exist always
end
if numel(ord)==1 %Scalar given, assuming it is the order of estimation
    if mod(ord,2)==0
      ord=ord+1;
    end
    weights=ones(1,ord);
  else
    %Assuming ord is the weight vector
    weights=ord;
    ord=numel(weights);
end

Xpp=0;
for k=1:ord
  Xpp= Xpp+weights(ord-k+1)*X(:,(k+1):(end-ord+k));
end

An=Xpp/X(:,1:end-ord); %Estimates w(ord)*A+w(ord-1)*A^2+...+w(1)*A^ord
A=unfoldMatrix(An,weights);
end
