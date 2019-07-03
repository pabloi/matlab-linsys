function [ii,I,ip,Ip]=trueStatInfoFilter(Y,CtRinvC,A,Q,BU,i0,I0,slowSamples)

[D2,N]=size(Y);
if nargin<8 || isempty(slowSamples)
    slowSamples=N;
end
D1=size(A,1);
previ=i0;
prevI=I0;

%Pre-comp for efficiency:
ey=eye(size(Q));
[ciQ,~,iQ]=pinvchol2(Q);
invertibleQ=norm(Q*iQ-ey,'fro')<1e-9;
iA=pinv(A); %This is order-of-magnitude slower than inv(A)
invertibleA=norm(A*iA-ey,'fro')<1e-9;
iQA=iQ*A;
ciQA=ciQ'*A;
AtiQA=ciQA'*ciQA;


%Init arrays:
ip=nan(D1,N+1);      ii=nan(D1,N);
Ip=nan(D1,D1,N+1);   I=nan(D1,D1,N);
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

  %Then, predict next step:
  if invertibleQ
      if invertibleA
          cI=mycholcov(prevI); %Could use chol() if prevI was guaranteed to be invertible, which has to be the case for invertibleA and after processing enough non-nan samples.
          iAcI=iA'*cI';
          %r=size(iAcI,2); %cI may be underrank
          [cP]=chol(iQ+iAcI*iAcI'); %Should always exist in this case
          iAicP=iA/cP;
          %icPt=cP\ey; %Should equal chol(inv(cP'*cP))
          b=cI*iAicP; %iAcI'*icPt; 
          iAcIb=iAcI*b;
          prevI=iAcI*iAcI' - iAcIb*iAcIb';
          previ=iQA*(iAicP*iAicP')*previ+prevI*BU(:,i); %=iQ*A*(iAicP*iAicP')
      else %prevI+A'*iQ*A needs to be invertible? Or can this be computed even on the PSD case?
          [cholAuxP,~,auxP]=pinvchol(prevI+AtiQA); %Needs pinvchol2?
          HH=(ciQA*cholAuxP);
          prevI=ciQ*(ey - HH*HH')*ciQ'; %If prevI is not invertible, this may result in a non-PSD matrix because of numerical issues
          previ=iQA*auxP*previ+prevI*BU(:,i);
      end
  elseif all(Q(:)==0)
      if invertibleA
          cI=mycholcov(prevI); %Could use chol() if prevI was guaranteed to be invertible, which has to be the case for invertibleA and after processing enough non-nan samples.
          iAcI=iA'*cI';
          prevI=iAcI*iAcI';
          previ=iA'*previ+prevI*BU(:,i);
      else %Some info will go to infinity! (0 uncertainty along some state subspace, numerically ugly)
          [x,P]=info2state(previ,prevI); %Requires one pinvchol2
          [previ,prevI]=state2info(A*x+BU(:,i),A*P*A'+Q); %Another pinvchol2
      end
  else %Q is PSD, unclear if the matrix inversion lemma can be used in any appropriate way, but I suspect that it can if prevI+AtiQA is invertible, and maybe even if it is not
      %This is the defeatist approach, requires 2 full pinvchol2 to handle all possible situations, which is very slow: 
      [x,P]=info2state(previ,prevI); %Requires one pinvchol2
      [previ,prevI]=state2info(A*x+BU(:,i),A*P*A'+Q); %Another pinvchol2
      
      %An alternative approach is to force Q to a barely-invertible case, and use the matrix inversion lemma:
      warning('trueStatInfoFilter:nonInvQ','Q was not invertible but improper priors were given. Regularizing Q for invertibility')
      %This is a workaround. The other alternative is to set improper priors
      %to large but finite values, so we do not need the inverse of Q.
      Q=Q+1e-7*eye(size(Q));
      invertibleQ=true;
      [ciQ,~,iQ]=pinvchol(Q);
      iQA=iQ*A;
      ciQA=ciQ'*A;
      AtiQA=ciQA'*ciQA;
  end
  if nargout>2 %Store Xp, Pp if requested:
      ip(:,i+1)=previ;   Ip(:,:,i+1)=prevI;
  end
end

%Do fast update:
%Get steady-state values:
oldI=prevI+CtRinvC; %Updated value
%Invert (not always available precomputed)
%[~,prevP]=info2state(previ,prevI); %Requires one pinvchol
[~,oldP]=info2state(previ,oldI); %Requires one pinvchol
%Store:
Ip(:,:,slowSamples+1:N)=repmat(prevI,1,1,N-slowSamples);
I(:,:,slowSamples+1:N)=repmat(oldI,1,1,N-slowSamples);

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
  end
end

end
