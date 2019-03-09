function [ii,I,ip,Ip,xf,Pf,xp,Pp]=trueStatInfoFilter(Y,CtRinvC,A,Q,BU,i0,I0,slowSamples)

[D2,N]=size(Y);
if nargin<8 || isempty(slowSamples)
    slowSamples=N;
end
D1=size(A,1);
previ=i0;
prevI=I0;

%Pre-comp for efficiency:
invertibleQ=all(diag(Q)>0);
[ciQ,~,iQ]=pinvchol(Q);
[~,~,iA]=pinvchol(A);
iQA=iQ*A;
ciQA=ciQ'*A;
AtiQA=ciQA'*ciQA;
ey=eye(size(Q));

%Init arrays:
if isa(Y,'gpuArray') %For code to work on gpu
    ip=nan(D1,N+1,'gpuArray');      ii=nan(D1,N,'gpuArray');
    Ip=nan(D1,D1,N+1,'gpuArray');   I=nan(D1,D1,N,'gpuArray');
else
    ip=nan(D1,N+1);      ii=nan(D1,N);
    Ip=nan(D1,D1,N+1);   I=nan(D1,D1,N);
    if nargout>4
        xp=zeros(D1,N+1);      xf=zeros(D1,N);
        Pp=zeros(D1,D1,N+1);   Pf=zeros(D1,D1,N);
    end
end
ip(:,1)=previ;
Ip(:,:,1)=prevI;


%Do traditional update-predict
psdEnforce=nargout>4 || ~invertibleQ; %IF user asks for x,P estimates, or if Q is not invertible, we need to do the update in state space
for i=1:slowSamples
  y=Y(:,i); %Output at this step
  %First, do the update given the output at this step:
  if ~any(isnan(y)) %If measurement is NaN, skip update.
     %[~,thisI,prevX,prevP,logL(i),rejSamples(i),prevI]=infoUpdate(CtRinvC,y,prevX,prevP,[],logDetCRC,invCRC);
       prevI=prevI + CtRinvC;
       previ=previ + y;
  end
  ii(:,i)=previ;
  I(:,:,i)=prevI; %Store results

  %Then, predict next step: (if prevI==0 there is no need for this)
  properPriors=diag(prevI)~=0;
  if all(properPriors) && ~psdEnforce  
      %State conversion not requested, all Info well defined, doing an efficient prediction
      [cholAuxP,~,auxP]=pinvchol(prevI+AtiQA);
      HH=(ciQA*cholAuxP);
      prevI=ciQ*(ey - HH*HH')*ciQ'; %I think this expression is better conditioned
      previ=iQA*auxP*previ+prevI*BU(:,i);
  elseif any(properPriors)
      %At least some info available: doing alternative way, which: 
      %guarantees PSD, works with non-inv Q, and as a bonus computes P and x
      %Con: if C'*inv(R)*C is under-rank (e.g. when dim of obs is less than
      %dim of state), the inverse of prevI will contain infinite elements
      %and this may be crap.
      [cOldP,~,oldP]=pinvchol(prevI);
      oldX=oldP*previ;
      AcP=A*cOldP;
      prevP=AcP*AcP'+Q;
      aux=find(~properPriors);
      prevP(aux,:)=0;
      prevP(:,aux)=0;
      prevP(sub2ind([D1,D1],aux,aux))=Inf;
      prevX=A*oldX+BU(:,i);
      [previ,prevI]=state2info(prevX,prevP); %Requires one pinvchol
  else %No info available: nothing to do, nowhere to go-o-o
      %I wanna be sedated
  end

  if nargout>2 %Store Xp, Pp if requested:
      ip(:,i+1)=previ;   Ip(:,:,i+1)=prevI;
      if nargout>4
          xp(:,i+1)=prevX;   Pp(:,:,i+1)=prevP;
          xf(:,i)=oldX;      Pf(:,:,i)=oldP;
      end
  end
end

%Do fast update:
%Get steady-state values:
oldI=prevI+CtRinvC; %Updated value
%Invert (not always available precomputed)
[~,prevP]=info2state(previ,prevI); %Requires one pinvchol
[~,oldP]=info2state(previ,oldI); %Requires one pinvchol
%Store:
Ip(:,:,slowSamples+1:N)=repmat(prevI,1,1,N-slowSamples);
I(:,:,slowSamples+1:N)=repmat(oldI,1,1,N-slowSamples);
if nargout>4
    Pp(:,:,slowSamples+1:N)=repmat(prevP,1,1,N-slowSamples);
    Pf(:,:,slowSamples+1:N)=repmat(oldP,1,1,N-slowSamples);
end
  
for i=(slowSamples+1):N %Update states succesively
  y=Y(:,i); %Output at this step
  %Update:
  if ~any(isnan(y))
    previ=previ+y;
  else
      %Should issue warning here: NaN and fast mode do not mix well
  end
  ii(:,i)=previ;
  
  %Predict:
  oldX=oldP*previ;
  prevX=A*oldX+BU(:,i);
  previ=prevI*prevX;

  if nargout>2 %Store Xp, Pp if requested:
      ip(:,i+1)=previ;  
      if nargout>4
          xp(:,i+1)=prevX;  
          xf(:,i)=oldX;   
      end
  end
end

end
