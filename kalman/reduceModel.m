function [Cnew,Rnew,Ynew,cRnew,logLmargin,logDetMargin,dimMargin,z2Margin]=reduceModel(C,R,Y)
  [D2,D1]=size(C);
  [icR,cR]=pinvchol(R); %This works if R is semidefinite, but in general
  %semidefinite R is unworkable, as R+C*P*C' needs to be invertible.
  %Even assuming P invertible at each update, it still requires R to be
  %invertible for all vectors orthogonal to the span of C at least)
  %Second, reduce the dimensionality problem:
  J=C'*icR; %Cholesky-like decomp of C'*inv(R)*C
  Yaux=icR'*Y;
  Rnew=J*J'; Ynew=J*Yaux; Cnew=Rnew;
  %cRnew=mycholcov(R);
  [icRnew,cRnew]=pinvchol(Rnew);

  %Now, compute factors to correct the estimated log-likelihood when it is computed from the reduced model:
  logDetMargin=sum(log(diag(cRnew)))-sum(log(diag(cR)));
  dimMargin=D2-D1; %Difference of data dim
  aux=icRnew'*Ynew;
  z2Margin=sum(Yaux.^2,1)-sum(aux.^2,1); %Difference of scores using R vs Rnew
  %Fun-fact: if R is optimized (as is in EM), then we expect the z2Margin to be roughly (exactly?): mean(z2Margin) =D2-D1. This is because the residuals along dimensions that get reduced can be used to design the optimal R given the residuals, and such value corresponds exacty to samples being on average 1-std away from the mean
  log2Pi=1.83787706640934529;
  logLmargin=-.5*(z2Margin +dimMargin*log2Pi)+logDetMargin; %Total difference between logL computed with the original vs. the reduced model.

  %Rationale: The matrix determinant lemma matrix allows us to show that for any P:
  %det(R+C'*P*C)*det(R) = det(Rnew + Cnew*P*Cnew')*det(Rnew)
  %Thus, the difference between log(det(R+C'*P*C)) with the original and reduceModel
  %values is: deltaLog = log(det(R))-log(det(Rnew)). This permits to compute the logL
  %in the kalman filter on the reduced model, and then correct it by adding a constant
  %term, instead of recomputing it.
  %Further, we can show that: (using Matrix Inversion Lemma)
  %y'*inv(R+C*P*C')*y = y'*inv(Rnew+Cnew*P*Cnew')*y -(y'*inv(R)*y - ynew'*inv(Rnew)*ynew)
end
