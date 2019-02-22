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
    I=eye(size(J));
    if nargin>7
        scale=(I-J)\(I-J^N)*K(:,1);
    else
        scale=(I-J)\K(:,1);
    end
    scale(isnan(scale) | isinf(scale) | scale==0)=1; %No way to scale states with 0 corresponding input, or for integrator scales (A_ii=1), leave as is
    V2=diag(1./scale);
    V=V2/V;
case 'canonicalAlt'
    %Same as canonical, but scaling but norm of columns of C
    [V,J]=diagonalizeA(A); %Diagonalizing A matrix
    [J,K]=transform(inv(V),A,B);
    scale=sqrt(sum(C.^2,1)).*sign(max(K,[],2))';
    scale(scale==0)=1; %Otherwise the transform is ill-defined
    V2=diag(scale);
    V=V2/V;
case 'orthonormal' %Orthonormalizing the columns of C
    [U,D,V]=svd(C);
    V=(D(1:size(C,2),:))*V;
    [J,K]=transform(V,A,B);
    I=eye(size(J));
    if nargin>7
        scale=(I-J)\(I-J^N)*K(:,1);
    else
        scale=(I-J)\K(:,1);
    end
    V=V*diag(sign(scale)); %So all states increase under a step input in first input
    case 'eyeQ'
        %To do: rotate such that Q is the identity matrix, except for any
        %zero diagonal entries it may have
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

function [V,J]=diagonalizeA(A)
    [V,J]=eig(A);
    % Deal with complex solutions, if they happen:
    a=imag(diag(J)); b=real(diag(J));
    if any(abs(a./b)>1e-15) %If there are (truly) complex eigen-values, will transform to the real-jordan form
        [V,J] = cdf2rdf(V,J);
        else %Ignore imaginary parts
        V=real(V);
        J=real(J);
    end
    % Sort states by decay rates: (these are only the decay rates if J is diagonal)
    [~,idx]=sort(diag(J)); %This works if the matrix is diagonalizable
    J=J(idx,idx);
    V=V(:,idx);
    
    %Alt: use block diagonal Schur
    %[V,J]=bdschur(A);
end
