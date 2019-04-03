function [A,B,C,D,Q,R,x0,P0]=initEM(Y,U,X,opts,P,Pt)
  %Initialization of parameters for EM-search
  %INPUTS:
  %Y= output data of linear system (My x N)
  %U= input data of linear system (Mu x N)
  %X= either initial guess of states (d x N matrix), or scalar stating the dimension of the state vector (d)

  if isempty(X)
      error('Xguess has to be a guess of the states (D x N matrix) or a scalar indicating the number of states to be estimated')
  elseif numel(X)==1 %X is just dimension, initializing as usual
      d=X;
      if any(isnan(Y(:))) %Removing NaNs first
        Y2=substituteNaNs(Y')';
      else
        Y2=Y;
      end
      %[A,B,C,D,X,Q,R]=subspaceIDv2(Y2,U,d); %Works if no missing data, is slow
      [X]=initGuessOld(Y2,U,d,opts);
  end
  if nargin<5
      P=[];
  end
  if nargin<6
      Pt=[];
  end
  [A,B,C,D,Q,R,x0,P0]=initParams(Y,U,X,opts,P,Pt);
end


function [A1,B1,C1,D1,Q1,R1,x01,P01]=initParams(Y,U,X,opts,Pguess,Ptguess)
  if isa(Y,'cell')
      [P,Pt]=cellfun(@(x,u,p) initCov(x,u,p),X,U,Pguess,'UniformOutput',false);
  else
      %Initialize covariance to plausible values:
      [P,Pt]=initCov(X,U,Pguess,Ptguess);

      %Move things to gpu if needed
      if isa(Y,'gpuArray')
          U=gpuArray(U);
          X=gpuArray(X);
          P=gpuArray(P);
          Pt=gpuArray(Pt);
      end
  end

%Initialize guesses of A,B,C,D,Q,R
[A1,B1,C1,D1,Q1,R1,x01,P01]=estimateParams(Y,U,X,P,Pt,opts);
%[A1,B1,C1,x01,~,Q1,P01] = canonize(A1,B1,C1,x01,Q1,P01,'canonicalAlt');
%UPDATE: Canonization is incompatible with fixed values for some parameters.
end

function [P,Pt]=initCov(X,U,P,Pt)
    [~,N]=size(X);
    %Initialize covariance to plausible values:
    if nargin<2 || isempty(P)
      dX=diff(X');
      if ~all(U(:)==0)
        dX=dX- U(:,1:end-1)'*(U(:,1:end-1)'\dX); %Projection orthogonal to input
      end
      Px=.1*(dX'*dX)/(N);
      P=repmat(Px,1,1,N);
      %Px1=(dX(2:end,:)'*dX(1:end-1,:));
      Pt=repmat(.2*diag(diag(Px)),1,1,N);
    elseif nargin<3 || isempty(Pt)
      Pt=.2*P;
    end
end

%This function used to be used instead of the subspace method
function [X]=initGuessOld(Y,U,D1,opts)
  if isa(Y,'cell')
      X=initGuessOld(cell2mat(Y),cell2mat(U),D1,opts);
      X=mat2cell(X,size(X,1),cellfun(@(x) size(x,2),Y));
  else
      idx=~any(isnan(Y),1);
      if ~isempty(opts.fixD) %If D was provided, no need to estimate
        D=opts.fixD;
      else
        D=Y(:,idx)/U(:,idx);
      end
      if isa(Y,'gpuArray')
          [pp,~,~]=pca(gather(Y(:,idx)-D*U(:,idx)),'Centered','off'); %Can this be done in the gpu?
      else
         [pp,~,~]=pca((Y(:,idx)-D*U(:,idx)),'Centered','off'); %Can this be done in the gpu?
      end
      X=1e-5*randn(D1,size(Y,2));
      if D1<=size(Y,1) %Finding less states than there are dimensions of output
          X(:,idx)=pp(:,1:D1)';
      else
          X(1:size(Y,1),idx)=pp';
      end
      X(:,~idx)=interp1(find(idx),X',find(~idx))';
      X=(1e2*X)./sqrt(sum(X.^2,2)); %Making sure we have good scaling, WLOG
  end
end
