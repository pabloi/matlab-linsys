function [CDrot,XUrot]=rotateFac(CD,XU,method,gamma)
  %Given the factorization of a matrix Y=CD*XU, this function rotates the components
  %such that the product is preserved, but some additional constraints are satisfied for the rotateFactors
  %Useful for rotation of C,D matrices and states of the output equation of a linear dynamical system
if nargin<3
  method='orthonormal';
end
if nargin<4
  gamma=1;
end
%Orthonormalizing first:
Y=CD*XU;
if ~strcmp(method,'none')
r=size(CD,2);
[U,D,V]=svd(Y);
CD=U(:,1:r); %Orthonormal
XU=D(1:r,1:r)*V(:,1:r)';
end
  switch method
  case 'orthonormal' %PCA style rotation: columns of CD will be orthonormal, and presented in decreasing order of variance explained of the original matrix
    CDrot=CD;
    XUrot=XU;
    case 'orthomax'
      [CDrot,T] = rotatefactors(CD, 'Method','orthomax','Coeff',gamma,'Maxit',1e3); %Uses gamma = 1 by default
      XUrot=T\XU;
      scale=sqrt(sum(XUrot.^2,2));
      [~,idx]=sort(scale,'descend');
      CDrot=CDrot(:,idx);
      XUrot=XUrot(idx,:);
    case 'varimax'
      [CDrot,XUrot]=rotateFac(CD,XU,'orthomax',1);
    case 'quartimax'
      [CDrot,XUrot]=rotateFac(CD,XU,'orthomax',0);
    case 'pablo'
      %scale=sqrt(sum(XU.^2,2));
      %XU=XU./scale;
      %CD=CD.*scale';
      [XUrot,T] = rotatefactors(XU', 'Method','orthomax','Coeff',1,'Maxit',1e3); %Uses gamma = 1 by default
      XUrot=XUrot';
      CDrot=CD/T';
      scale=sqrt(sum(CDrot.^2,1));
      CDrot=CDrot./scale;
      XUrot=scale'.*XUrot;
      scale=sqrt(sum(XUrot.^2,2));
      [~,idx]=sort(scale,'descend');
      CDrot=CDrot(:,idx);
      XUrot=XUrot(idx,:);
    case 'none'
      CDrot=CD;
      XUrot=XU;
    otherwise
      warning('Unrecognized rotation method, orthonormalizing.')
      CDrot=CD;
      XUrot=XU;
  end
sum(CDrot.^2,1)
norm(CDrot'*CDrot-eye(size(CD,2)),'fro')
sqrt(sum(XUrot.^2,2))
norm(Y-CDrot*XUrot,'fro')
