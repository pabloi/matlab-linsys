function [ii,I,ip,Ip]=trueStatInfoFilter(Y,CtRinvC,A,Q,BU,i0,I0,slowSamples)

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
  if invertibleQ %Also requires invertible prevI+A*inv(Q)*A'
      [cholAuxP,~,auxP]=pinvchol(prevI+AtiQA); %This may be problematic if Q has some 0 variances where we have 0 info.
      HH=(ciQA*cholAuxP);
      prevI=ciQ*(ey - HH*HH')*ciQ'; %I think this expression is better conditioned
      previ=iQA*auxP*previ+prevI*BU(:,i);
  elseif invertibleA && all(Q(:)==0) %This can probably be made to work when not all Q==0, by restricting to differnt subspaces
      cI=mycholcov(prevI);
      iAcI=iA'*cI;
      prevI=iAcI*iAcI';
      previ=iA'*previ+prevI*BU(:,i);
  else %Not invertible Q, but some elements of Q are different from 0, or A is non-invertible
      [x,P]=info2state(previ,prevI); %Requires one pinvchol
      if all(all(abs(P*prevI-eye(size(prevI)))<1e-5)) %Invertible prevI: doing traditional update
         [previ,prevI]=state2info(A*x+BU(:,i),A*P*A'+Q); 
      else %Enforcing invertibility of Q through regularization
        warning('trueStatInfoFilter:nonInvQ','Q was not invertible but improper priors were given. Regularizing Q for invertibility')
        %This is a workaround. The other alternative is to set improper priors
        %to large but finite values, so we do not need the inverse of Q.
        %This workaround presumes that prevI+A*inv(Q)*A' is invertible
        Q=Q+1e-7*eye(size(Q));
        invertibleQ=true;
        [ciQ,~,iQ]=pinvchol(Q);
        %iQA=iQ*A;
        ciQA=ciQ'*A;
        AtiQA=ciQA'*ciQA;
        [cholAuxP,~,auxP]=pinvchol(prevI+AtiQA); %This may be problematic if Q has some 0 variances where we have 0 info.
        HH=(ciQA*cholAuxP);
        prevI=ciQ*(ey - HH*HH')*ciQ'; %I think this expression is better conditioned
        previ=iQA*auxP*previ+prevI*BU(:,i);
      end
  end

  if nargout>2 %Store Xp, Pp if requested:
      ip(:,i+1)=previ;   Ip(:,:,i+1)=prevI;
      if nargout>4
          error('!') %This is not computed properly
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
