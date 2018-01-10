function [model] = sPCAv7(Y,dynOrder,forcePCS,nullBD,outputUnderRank)
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
rankC=dynOrder-outputUnderRank;

%% Find a first solution: PCA + dynamic fit over PCA coefficients: fast and good enough
rankD=0;
if ~nullBD
    rankD=1;
end
rankCD=rankC+rankD;
[p,c]=pca(Y','Centered',false); %Do PCA to extract the #order most meaningful PCs:
CD=c(:,1:rankCD);
P=p(:,1:rankCD)';

%Estimate dynamics from PCA:
[J,X,V,K] = estimateDynv3(P, realPolesOnly, nullBD, dynOrder);
%EstimateDynv3 returns states X such that states are exponentially decaying
%(except constant term, if present) so asymptotic state is 0, and initial
%states are all =1.
CD=CD*[V K]; %Equivalent to: CD=(CD*P)/X; %Rotating PCs
%r2=1-norm(Y'-(CD*X),'fro')^2/norm(Y','fro')^2

if ~forcePCS
    %Iterate for optimal solution:
    ii=0;
    while ii<5 %For some reason this converges very quickly (more iterations return very fast, no cost really in having them)
    CD=Y'/X; %Compute optimal subspace given states trajectories
    %r2=1-norm(Y'-(CD*X),'fro')^2/norm(Y','fro')^2
    if outputUnderRank>0 && size(CD,1)>=rankC   %Need to reduce rank: using reduced-rank reg when dim data => M
        %See: https://stats.stackexchange.com/questions/152517/what-is-reduced-rank-regression-all-about
        Yfit=CD*X;
        [ww,hh,aa]=pca(Yfit','Centered',false);
        CD=ww(:,1:rankCD)*ww(:,1:rankCD)'*CD;
    end
    [J,X,V,K] = estimateDynv3(CD\Y', realPolesOnly, nullBD, J); % Compute optimal states given projection onto subspace
    CD=CD*[V K]; %Re-expressing subspace basis in canonically-decoupled states
    %r2=1-norm(Y'-(CD*X),'fro')^2/norm(Y','fro')^2
    ii=ii+1;
    end
end

%% Decompose solution:
C=CD(:,1:rankC);
X=X(1:rankC,:);
D=CD(:,[rankC+1:end]); %Empty if nullBD==true
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
r2=1-norm(Y'-([C D]*[X; ones(size(D,2),size(X,2))]),'fro')^2/norm(Y','fro')^2;

%% Assign outputs:
model.C=C;
model.J=J;
model.X=X;
model.B=B;
model.D=D;
model.r2=r2;
end