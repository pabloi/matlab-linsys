function [Xs,Ps,X,P,Xp,Pp,rejSamples]=statKalmanSmoother(Y,A,C,Q,R,x0,P0,B,D,U,outRejFlag,constFun)

  %Init missing params:
  if nargin<6 || isempty(x0)
    x0=zeros(size(A,1),1); %Column vector
  end
  if nargin<7 || isempty(P0)
    P0=1e8 * eye(size(A));
  end
  if nargin<8 || isempty(B)
    B=0;
  end
  if nargin<9 || isempty(D)
    D=0;
  end
  if nargin<10 || isempty(U)
     U=zeros(1,size(X,2)); 
  end
  if nargin<11 || isempty(outRejFlag)
      outRejFlag=0;
  end
  if nargin<12 || isempty(constFun)
      constFunFlag=0;
  else
      constFunFlag=1;
  end

  %Size checks:
  %TODO

%Step 1: forward filter
if constFunFlag==0
    %[X,P,Xp,Pp,rejSamples]=filterStationary(Y,A,C,Q,R,x0,P0,B,D,U,outRejFlag);
    [X,P,Xp,Pp,rejSamples]=statKalmanFilter(Y,A,C,Q,R,x0,P0,B,D,U,outRejFlag);
else
    [X,P,Xp,Pp,rejSamples]=filterStationary_wConstraint(Y,A,C,Q,R,x0,P0,B,D,U,constFun);  
end

%Step 2: backward pass: (following the Rauch-Tung-Striebel implementation:
%https://en.wikipedia.org/wiki/Kalman_filter#Fixed-interval_smoothers)
Xs=X;
Ps=P;
prevXs=X(:,end);
prevPs=P(:,:,end);
S=pinv(Q)*A;

for i=(size(Y,2)-1):-1:1
  %H= pinv(P(:,:,i)) + A'*S;
  %invH=pinv(H);
  %newK=invH*S';
  %Equivalent to:
  PP=P(:,:,i);
  prevPriorP=Pp(:,:,i+1);
  newK=PP*A'/prevPriorP; %Pp=A*P*A'+Q, so A*newK = I -Q/Pp
  x=X(:,i);
  prevXs=x + newK*(prevXs-A*x);
  Xs(:,i)=prevXs;
  %prevPs=invH + newK*pinv(prevPs)*newK';
  %prevPs=newK/S' + (newK/prevPs)*newK';
  prevPs=PP + newK*(prevPs - prevPriorP)*newK';
  Ps(:,:,i)=prevPs;
end

end
