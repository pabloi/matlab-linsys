function [X,P,Xp,Pp,logL,rejSamples,icS] = trueKF(C,R,A,Q,Y_D,BU,prevX,prevP,rejTh)
M=size(Y_D,2);
X=zeros(size(prevX,1),M);
P=zeros(size(prevX,1),size(prevX,1),M);
Xp=X;
Pp=P;
logL=zeros(M,1);
rejSamples=false(M,1);
icS=R; %doxy
for i=1:M
  y=Y_D(:,i); %Output at this step

  %First, do the update given the output at this step:
  if ~any(isnan(y)) %If measurement is NaN, skip update.
     [prevX,prevP,logL(i),rejSamples(i),icS]=KFupdate2(C,R,y,prevX,prevP,rejTh);
  end
  X(:,i)=prevX;  P(:,:,i)=prevP; %Store results

  %Then, predict next step:
  [prevX,prevP]=KFpredict(A,Q,prevX,prevP,BU(:,i));
  if nargout>2 %Store Xp, Pp if requested:
      Xp(:,i)=prevX;   Pp(:,:,i)=prevP; 
  end
end
end

