function [A,B,C,D,Q,R,x0,P0]=estimateParams(Y,U,X,P,Pt,robustFlag)
%M-step of EM estimation for LTI-SSM
%INPUT:
%Y = output of the system, D2 x N
%U = input of the system, D3 x N
%X = state estimates of the system (Kalman-smoothed), D1 x N
%P = covariance of states (Kalman-smoothed), D1 x D1 x N
%Pp = covariance of state transitions (Kalman-smoothed), D1 x D1 x (N-1),
%evaluated at k+1|k
%See Cheng and Sabes 2006, Ghahramani and Hinton 1996, Shumway and Stoffer 1982

%
if nargin<6 || isempty(robustFlag)
    robustFlag=false;
end
%Define vars:
[yx,yu,xx,uu,xu,SP,SPt,xx_,uu_,xu_,xx1,xu1,SP_,S_P]=computeRelevantMatrices(Y,X,U,P,Pt);
D1=size(xx,1);

%Estimate x0,P0:
if isa(X,'cell')
    [x0,P0]=cellfun(@(x,p) estimateInit(x,p),X,P,'UniformOutput',false);
else
    [x0,P0]=estimateInit(X,P);
end

%Estimate A,B:
O=[SP_+xx_ xu_; xu_' uu_];
AB=[SPt+xx1 xu1]/O; %In absence of uncertainty, reduces to: [A,B]=X+/[X;U],
%where X+ is X one step in the future
A=AB(:,1:D1);
B=AB(:,D1+1:end);

%Estimate C,D:
O=[SP+xx xu; xu' uu];
CD=[yx,yu]/O; %Notice that in absence of uncertainty in states, this reduces to [C,D]=Y/[X;U]
C=CD(:,1:D1);
D=CD(:,D1+1:end);

%Estimate Q,R: %Adaptation of Shumway and Stoffer 1982: (there B=D=0 and C is fixed), but consistent with Ghahramani and Hinton 1996, and Cheng and Sabes 2006
[w,z]=computeResiduals(Y,U,X,A,B,C,D);

% MLE estimator of Q, under the given assumptions:
aux=mycholcov(SP_); %Enforce symmetry
Aa=A*aux';
Nw=size(w,2);
APt=A*SPt';
Q2=(S_P-(APt+APt')+Aa*Aa')/(Nw);
%Q2=(S_P-A*SPt')/Nw; %This is equivalent to the expression above if these 
%matrices come from kalman smoothing.
sQ=mycholcov(Q2);
Q2=sQ'*sQ; %Enforcing psd, unclear if necessary
%See Ghahramani and Hinton, and Cheng and Sabes

if ~robustFlag
    Q1=(w*w')/(Nw); 
%Covariance of EXPECTED residuals given the data and params
%not designed to deal with outliers, autocorrelated w
%Note: if we dont have exact extimates of A,B, then the residuals w are not
%iid gaussian. They will be autocorrelated AND have outliers with respect
%to the best-fitting multivariate normal. Thus, we benefit from doing a
%more robust estimate, especially to avoid local minima in trueEM
else
%Robust estimation manages to avoid the failure mode where Q is overestimated 
%because of the presence of 'outlier' observations (which may or may not be
%true outliers), which in turn causes predicted states to be very uncertain, 
%which causes large state updates when those 'outliers' are observed, leading 
%to large state residuals w, which leads to large Q and so on.
%It also breaks the non-decreasing logL guarantee of EM, and slightly
%slower than the classical update: 10ms per call on my current setup, which
%is 100+% the time of the whole estimateParams(). However, improved
%parameter estimation may lead to faster overall EM() running time if the
%fastFlag is enabled.
    Q1=robCov(w); %Fast variant of robustcov() estimation
end
Q=Q1 +Q2;

%MLE of R:
aux=mycholcov(SP); %Enforce symmetry
Ca=C*aux';
Nz=size(z,2);
if ~robustFlag
    R1=(z*z')/Nz;
else
    R1=robCov(z);
end
R2=(Ca*Ca')/Nz;
R=R1+R2;
%R=(z*Y')/Nz; %Equivalent to above, but does not enforce symmetry
R=R+1e-15*eye(size(R)); %Avoid numerical issues


end

function [x0,P0]=estimateInit(X,P,A,Q)
x0=X(:,1); %Smoothed estimate
P0=P(:,:,1); %Smoothed estimate, the problem with this estimate is that it is monotonically decreasing on the iteration of trueEM(). More likely it should converge to the same prior uncertainty we have for all other states.
%A variant to not make it monotonically decreasing:
%Aa=A*aux';
%P0=Q+Aa*Aa';
end

function [yx,yu,xx,uu,xu,SP,SPt,xx_,uu_,xu_,xx1,xu1,SP_,S_P]=computeRelevantMatrices(Y,X,U,P,Pt)
%Notice all outputs are DxD matrices, where D=size(X,1);

if isa(X,'cell') %Case where data is many realizations of same system
    [yx,yu,xx,uu,xu,SP,SPt,xx_,uu_,xu_,xx1,xu1,SP_,S_P]=computeRelevantMatrices(Y{1},X{1},U{1},P{1},Pt{1});
    for i=2:numel(X)
        [yxa,yua,xxa,uua,xua,SPa,SPta,xx_a,uu_a,xu_a,xx1a,xu1a,SP_a,S_Pa]=computeRelevantMatrices(Y{i},X{i},U{i},P{i},Pt{i});
        xx=xx+xxa;
        xu=xu+xua;
        uu=uu+uua;
        xx_=xx_+xx_a;
        xu_=xu_+xu_a;
        uu_=uu_+uu_a;
        xx1=xx1+xx1a;
        xu1=xu1+xu1a;
        SP=SP+SPa;
        SP_=SP_+SP_a;
        S_P=S_P+S_Pa;
        SPt=SPt+SPta;
        yx=yx+yxa;
        yu=yu+yua;
    end
else %Data is in matrix form, i.e., single realization
    %Data for A,B estimation:
    %xu=X*U';
    xu_=X(:,1:end-1)*U(:,1:end-1)'; %=xu - X(:,end)*U(:,end)'
    %uu=U*U';
    uu_=U(:,1:end-1)*U(:,1:end-1)'; %=uu - U(:,end)*U(:,end)'
    %xx=X*X';
    xx_=X(:,1:end-1)*X(:,1:end-1)'; %=xx - X(:,end)*X(:,end)'
    %SP=sum(P,3);
    SP_=sum(P(:,:,1:end-1),3); %=SP-P(:,:,end);
    S_P=sum(P(:,:,2:end),3); %=SP-P(:,:,1);
    SPt=sum(Pt,3);
    xu1=X(:,2:end)*U(:,1:end-1)';
    xx1=X(:,2:end)*X(:,1:end-1)';
    %Remove data associated to NaN values:
    if any(any(isnan(Y)))
      idx=~any(isnan(Y));
      Y=Y(:,idx);
      X=X(:,idx);
      U=U(:,idx);
      P=P(:,:,idx);
    end
    %Data for C,D estimation:
    SP=sum(P,3);
    xu=X*U';
    uu=U*U';
    xx=X*X';
    yx=Y*X';
    yu=Y*U';
end

end

function [w,z]=computeResiduals(Y,U,X,A,B,C,D)

if isa(X,'cell') %Case where data is many realizations of same system
    [w,z]=cellfun(@(y,u,x) computeResiduals(y,u,x,A,B,C,D),Y(:),U(:),X(:),'UniformOutput',false); %Ensures column cell-array output
    w=cell2mat(w'); %Concatenates as if each realization is extra samples
    z=cell2mat(z');
else
    N=size(X,2);
    idx=~any(isnan(Y));
    z=Y-C*X-D*U;
    z=z(:,idx);
    w=X(:,2:N)-A*X(:,1:N-1)-B*U(:,1:N-1);
end

end
