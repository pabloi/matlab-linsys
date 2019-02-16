function [ii,I,ip,Ip,xf,Pf,xp,Pp]=trueStatInfoFilter(Y,CtRinvC,A,Q,BU,i0,I0,slowSamples)

[D2,N]=size(Y);
if nargin<8 || isempty(slowSamples)
    slowSamples=N;
end
D1=size(A,1);
previ=i0;
prevI=I0;

%Pre-comp for efficiency:
[ciQ,~,iQ]=pinvchol(Q);
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
        xp=nan(D1,N+1);      xf=nan(D1,N);
        Pp=nan(D1,D1,N+1);   Pf=nan(D1,D1,N);
    end
end
ip(:,1)=previ;
Ip(:,:,1)=prevI;


%Do traditional update-predict
psdEnforce=nargout>4;
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
  if all(diag(prevI)~=0) && ~psdEnforce 
      %State conversion not requested, all Info well defined, doing an efficient prediction
      [cholAuxP,~,auxP]=pinvchol(prevI+AtiQA);
      HH=(ciQA*cholAuxP);
      prevI=ciQ*(ey - HH*HH')*ciQ'; %I think this expression is better conditioned
      previ=iQA*auxP*previ+prevI*BU(:,i);
  elseif any(diag(prevI)~=0) %We have SOME info: doing alternative way, 
      %which guarantees PSD, and as a bonus computes P and x
      [oldX,oldP]=info2state(previ,prevI); %Requires one pinvchol
      prevP=A*oldP*A'+Q;
      prevX=A*oldX+BU(:,i);
      [previ,prevI]=state2info(prevX,prevP); %Requires one pinvchol
  else %information matrix all 0, nothing to update
        oldX=zeros(size(previ));
        oldP=diag(Inf*ones(size(previ)));
        prevX=zeros(size(previ));
        prevP=diag(Inf*ones(size(previ)));
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
