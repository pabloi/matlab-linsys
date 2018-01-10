function [C,J,X,B,D,r2,V] = sPCAv2(Y,order,forcePCS,nullBD)
%sPCA or smooth PCA, aims at estimating a best-fit space-state model from
%given outputs Y, and assuming constant input. It is similar to cPCA, but
%it doesn't purposefully identify the canonical states
%It returns the a (best in some sense?) fit of the form Y'~C*X + B ; with X(i+1,:)=A*X(i,:)+D
%where C are the first #order PCs from PCA, and A is a matrix with strictly real 
%& different eigen-vectors (no complex or double-pole solutions allowed)
%X is scaled such that X(0)=1 for all states.
%INPUTS:
%Y = N x D data matrix, representing N samples of d-dimensional data
%order: number of principal components to be estimated
%forcePCS: flag to indicate if the solution is constrained to be a linear transformation of the PCA subspace determined by first #order PCs
%OUTPUTS:
%C: D x order matrix, representing map from states to output (Y) minus constant (D)
%A: evolution matrix for states, such that X(:,i+1)=A*X(:,i)+B
%X: smoothed state estimators
%B:
%D:
%r2: r^2 of data to reconstruction
%V: transformation from PCA's PCs to canonic states (only makes sense if forcePCS=true)
%See also: sPCA_knownYinf

% Pablo A. Iturralde - Univ. of Pittsburgh - Last rev: Jun 14th 2017 %Need
% to update description


if nargin<2 || isempty(order)
    order=2; %Minimum order for which this makes sense
end
if nargin<3 || isempty(forcePCS)
    forcePCS=false; %If true, this flag forces the columns of C to lie in the subspace spanned by the first #order PCs from PCA
end
if nargin<4 || isempty(nullBD)
    nullBD=false;
end

NN=size(Y,1); %Number of samples
DD=size(Y,2); %Dimensionality of data

%Do PCA to extract the #order most meaningful PCs:
mY=mean(Y,1);
mY=0;
[p,c,a]=pca((Y-mY)','Centered',false);
if ~nullBD
    CD=c(:,1:order+1);
    P=p(:,1:order+1)';
    r2pca=sum(a(1:order+1))/sum(a);
else
    CD=c(:,1:order);
    P=p(:,1:order);
    r2pca=sum(a(1:order))/sum(a)
end

%% Optimize to find best decaying exponential fits:
realPolesOnly=true;
if ~forcePCS
    maxIter=1; %Apparently iterating does NOT improve performance (in fact, it makes it worse)
    %Intuitively, it seems there has to be a convergence issue in estimateDyn.
    %Alternatively, it may be that the decoupling of the problem into PCA +
    %smooth dynamics only works appropriately for PCA space
    %**After making changes such that P always has orthogonal columns (to
    %avoid ill-conditioned situations that may be numerically hard)
    %performance is still the same or (very slightly) worse. This suggests
    %that orthogonality of P is needed for this to work well.**
else
    maxIter=1; %No need to iterate, since CD doesn't change
end
iter=0;
J=[];
while iter<maxIter
    iter=iter+1;
    [J,B,X,V] = estimateDyn(P, realPolesOnly, nullBD,J);
    if iter==1
        CD=(CD*P)/X;
    end
    if ~forcePCS
        CD=Y'/X; %This allows CD to escape the subspace spanned by the PCs from PCA, and improves r2 slightly
    end
    
    %This is only needed if we want to iterate:
    %[P,CD2,a]=pca(CD\Y','Centered',false);
    %P=P';
end

C=CD(:,1:order);
if ~nullBD
    D=CD(:,order+1);
    X=X(1:order,:);
else
    D=0;
end

%Reconstruction value:
rA=1-norm(X(:,2:end)-J*X(:,1:end-1)-B,'fro')^2/norm(Y','fro')^2 %This has to be 1
r2(1)=1-norm(Y'-(C*X+D),'fro')^2/norm(Y','fro')^2;
r2(2)=1-norm(Y'-(C*X+D),'fro')^2/norm(Y'-D,'fro')^2;
end

