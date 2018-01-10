function [C,J,X,B,D,r2] = sPCAv5(Y,dynOrder,forcePCS,nullBD,outputUnderRank)
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
%nullBD: flag that forces 
%OUTPUTS:
%C: D x order matrix, representing map from states to output (Y) minus constant (D)
%A: evolution matrix for states, such that X(:,i+1)=A*X(:,i)+B
%X: smoothed state estimators
%B:
%D:
%r2: r^2 of data to reconstruction
%See also: estimateDynv2.m

% Pablo A. Iturralde - Univ. of Pittsburgh - Last rev: Jun 27th 2017 %Need
% to update description


if nargin<2 || isempty(dynOrder)
    dynOrder=2; %Minimum order for which this makes sense
end
if nargin<3 || isempty(forcePCS)
    forcePCS=false; %If true, this flag forces the columns of C to lie in the subspace spanned by the first #order PCs from PCA
end
if nargin<4 || isempty(nullBD)
    nullBD=false;
end
if nargin<5 || isempty(outputUnderRank)
    outputUnderRank=0;
end

NN=size(Y,1); %Number of samples
DD=size(Y,2); %Dimensionality of data
realPolesOnly=true; %Only acceptable value right now

%% Find a first solution: PCA + dynamic fit over PCA coefficients: fast and good enough
%Do PCA to extract the #order most meaningful PCs:
if ~nullBD
    mY=mean(Y,1); %Better-conditioned problem(?) if we remove the mean first, but less optimal(?)
else
    mY=0;
end
M=dynOrder-outputUnderRank;
YY=(Y-mY)';
[p,c]=pca(YY,'Centered',false);
CD=c(:,1:M);
P=p(:,1:M)';
if ~nullBD
    CD=[CD mY'];
    P=[P;ones(1,size(P,2))];
    M=M+1;
end

%Estimate dynamics:
[J,X,V,K] = estimateDynv3(P, realPolesOnly, nullBD, dynOrder);
if forcePCS
    CD=CD*[V K]; %Equivalent to: CD=(CD*P)/X; %Rotating PCs
    if ~nullBD
        CD(:,dynOrder+1)=CD(:,dynOrder+1)+mY';
    end
else
    CD=Y'/X; %This improves r2 slightly (and may increase rank if outputUnderRank>0)
    if outputUnderRank>0 && size(CD,1)>=M   %Need to reduce rank: using reduced-rank reg when dim data => M
        %See: https://stats.stackexchange.com/questions/152517/what-is-reduced-rank-regression-all-about
        Yfit=CD*X;
        [ww,hh,aa]=pca(Yfit','Centered',false);
        CD=ww(:,1:M)*ww(:,1:M)'*CD;
    end
end

%% Optimize solution: %Estimating J,B,CD altogether, is slow and in practice improves very little (convergence issues?)
%  if ~forcePCS %No improvement possible if forcePCS=true
%      [J,X] = estimateDynv3(YY, realPolesOnly, nullBD, J);
%      CD=YY/X;
%  end

%% Decompose solution:
C=CD(:,1:dynOrder);
if ~nullBD
    D=CD(:,dynOrder+1);
    X=X(1:dynOrder,:);
else
    D=zeros(size(C,1),1);
end
B=zeros(size(X,1),1); %This is by convention of estimateDynv3 results
%% Normalize columns of C as convention (and normalize X accordingly)
scale=sqrt(sum(C.^2));
C=bsxfun(@rdivide,C,scale);
X=bsxfun(@times,X,scale');
%% Change initial states, as convention, when nullBD~=0 such that x(0)=0 and states grow
if ~nullBD
    [B,D,X]=chngInitState(J,B,C,D,X,zeros(size(X,1),1));
    B=-B;
    C=-C;
    X=-X;
end
%% Reconstruction values:
%rA=1-norm(X(:,2:end)-J*X(:,1:end-1)-B,'fro')^2/norm(Y','fro')^2 %This has to be exactly 1
r2=1-norm(Y'-(C*X+D),'fro')^2/norm(Y','fro')^2;
end