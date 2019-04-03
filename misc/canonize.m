function [J,B,C,X,V,Q,P] = canonize(A,B,C,X,Q,P,method,N)
    %General transformation of a linear model to give a unique representation.
%Input N is only used with method canonical, and is optional (if not given, presumed N=infinity)

if nargin<6 || isempty(P)
    P=zeros(size(A));
end
if nargin<5 || isempty(Q)
    Q=zeros(size(A));
end
if nargin<4 || isempty(X)
    X=zeros(size(A,1),1);
end
if nargin<7 || isempty(method)
    method='canonical';
end

switch method
case 'canonical'
    [V,J]=diagonalizeA(A); %Diagonalizing A matrix
    %% Scale so that states reach a value of 1 at some point in time under a step input in first input
    %The point in time is N if given, infinity otherwise
    [J,K]=transform(inv(V),A,B);
    if nargin<8
        N=[];
    end
    scale=xinf(J,K,N);
    scale(isnan(scale) | isinf(scale) | scale==0)=1; %No way to scale states with 0 corresponding input, or for integrator scales (A_ii=1), leave as is
    V2=diag(1./scale);
    V=V2/V;
case 'canonicalAlt'
    %Same as canonical, but scaling but norm of columns of C
    [V,~]=diagonalizeA(A); %Diagonalizing A matrix
    [~,K,C1]=transform(inv(V),A,B,C);
    idx=abs(K)==max(abs(K),[],2); %Location of max element in abs() per row 
    scale=sqrt(sum(C1.^2,1)).*sign(sum(K.*idx,2))';
    scale(scale==0)=1; %Otherwise the transform is ill-defined
    V2=diag(scale);
    V=V2/V;
case 'orthonormal' %Orthonormalizing the columns of C
    [U,D,V]=svd(C);
    V=(D(1:size(C,2),:))*V;
    [J,K]=transform(V,A,B);
    if nargin<8
        N=[];
    end
    scale=xinf(J,K,N);
    V=V*diag(sign(scale)); %So all states increase under a step input in first input
    case 'eyeQ'
        %To do: rotate such that Q is the identity matrix, except for any
        %zero diagonal entries it may have
        [V,~,~,L,D]=pinvchol2(Q);
        tol=1e-9;
        zeroStates= abs(diag(D))<tol;
        badStates=diag(D)<0 & ~zeroStates;
        if any(badStates)
            error('canonize:eyeQ','Q is not PSD (within tolerance), aborting.');
        elseif any(zeroStates)
            warning('canonize:eyeQ','Cannot make Q=I. Ignoring numerically-zero variance eigen-states.');
            V(isinf(V))=1; %Leaving state as is, should do something to scale it to some appropriate magnitude.
        end
        V=V';
        [J,K]=transform(V,A,B);
        scale=xinf(J,K);
        V=diag(sign(scale))*V;
    case 'diagQ'
        %Similar to eye Q, but just diagonalizing Q and scaling so columns
        %of C are normal
         [V,~,~,L,D]=pinvchol2(Q);
        tol=1e-9;
        zeroStates= abs(diag(D))<tol;
        badStates=diag(D)<0 & ~zeroStates;
        if any(badStates)
            error('canonize:eyeQ','Q is not PSD (within tolerance), aborting.');
        elseif any(zeroStates)
            warning('canonize:eyeQ','Cannot make Q=I. Ignoring numerically-zero variance eigen-states.');
            V(isinf(V))=1; %Leaving state as is, should do something to scale it to some appropriate magnitude.
        end
        V=V';
        [J,K,CC]=transform(V,A,B,C);
        scale=xinf(J,K);
        V=diag(sign(scale))*V;
        scale=sqrt(sum(CC.^2,1));
        V=diag(scale)*V;
    case 'orthomax'
      [Cr,V] = rotatefactors(C, 'Method','orthomax','Coeff',gamma,'Maxit',1e3); %Uses gamma = 1 by default
      scale=sqrt(sum(Cr.^2,1)).*sign(max(B,[],2))'; %Need to order with some criteria
      scale(scale==0)=1; %Otherwise the transform is ill-defined
      V2=diag(scale);
      V=V2/V;
     case 'varimax'
      [Cr,V] = rotatefactors(C, 'Method','orthomax','Coeff',1,'Maxit',1e3);
      scale=sqrt(sum(Cr.^2,1)).*sign(max(B,[],2))'; %Need to order with some criteria
      scale(scale==0)=1; %Otherwise the transform is ill-defined
      V2=diag(scale);
      V=V2/V;
     case 'quartimax'
      [Cr,V] = rotatefactors(C, 'Method','orthomax','Coeff',0,'Maxit',1e3);
      scale=sqrt(sum(Cr.^2,1)).*sign(max(B,[],2))'; %Need to order with some criteria
      scale(scale==0)=1; %Otherwise the transform is ill-defined
      V2=diag(scale);
      V=V2/V;
otherwise
    error('Unrecognized method')
end
%% Transform with all the changes:
[J,B,C,Q,X,P]=transform(V,A,B,C,Q,X,P);

end

function ss=xinf(A,B,N)
%Computes the steady-state value of x under a step-input on the first
%component
I=eye(size(A));
    if nargin>2 && ~isempty(N)
        ss=(I-A)\(I-A^N)*B(:,1);
    else
        ss=(I-A)\B(:,1);
    end
end
