function [A,B,C,D,Q,R,x0,P0]=estimateParams(Y,U,X,P,Pt,opts)
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
[No,N]=size(Y);
Nu=size(U,1);
%
[opts] = processEMopts(opts);
if isempty(opts.indD)
  opts.indD=1:Nu;
end
if isempty(opts.indB)
  opts.indB=1:Nu;
end

%
if all(U(:)==0)  %Can't estimate B,D if input is null/empty
    U=zeros(0,N);
end

%Define vars:
[yx,yu,xx,uu,xu,SP,SPt,xx_,uu_,xu_,xx1,xu1,SP_,S_P]=computeRelevantMatrices(Y,X,U,P,Pt,opts.robustFlag);
D1=size(xx,1);

%Estimate A,B:
O=[SP_+xx_ xu_; xu_' uu_];
AB=[SPt+xx1 xu1]/O; %In absence of uncertainty, reduces to: [A,B]=X+/[X;U],
%where X+ is X one step in the future
A=AB(:,1:D1);
B=AB(:,D1+1:end);

%Estimate C,D:
xu=xu(:,opts.indD);
uu=uu(opts.indD,opts.indD);
yu=yu(:,opts.indD);
O=[SP+xx xu; xu' uu];
CD=[yx,yu]/O; %Notice that in absence of uncertainty in states, this reduces to [C,D]=Y/[X;U]
C=CD(:,1:D1);
D=zeros(No,Nu);
D(:,opts.indD)=CD(:,D1+1:end);

if isempty(U)
    B=ones(D1,Nu); %Setting zeros here makes some canonical (e.g. canonizev2) forms ill-defined
    D=zeros(No,Nu);
    U=zeros(Nu,N);
end

%Estimate Q,R: %Adaptation of Shumway and Stoffer 1982: (there B=D=0 and C is fixed), but consistent with Ghahramani and Hinton 1996, and Cheng and Sabes 2006
[w,z]=computeResiduals(Y,U,X,A,B,C,D);

% MLE estimator of Q, under the given assumptions:
aux=mycholcov(SP_); %Enforce symmetry
Aa=A*aux';
Nw=size(w,2);
APt=A*SPt';
Q2=(S_P-(APt+APt')+Aa*Aa')/(Nw);
sQ=mycholcov(Q2);
Q2=sQ'*sQ; %Enforcing psd, unclear if necessary
%See Ghahramani and Hinton, and Cheng and Sabes

if ~opts.robustFlag
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
    [Q1]=robCov(w);%,95); %Fast variant of robustcov() estimation
    %Q1=squeeze(median(w.*reshape(w',1,size(w,2),size(w,1)),2));
end
Q=Q1 +Q2;

%MLE of R:
aux=mycholcov(SP); %Enforce symmetry
Ca=C*aux';
Nz=size(z,2);
%if ~robustFlag
    R1=(z*z')/Nz;
%else
%    R1=robCov(z,95); %Estimating R robustly leads to instabilities and bad
%    local maxima
%end
R2=(Ca*Ca')/Nz;
R=R1+R2;
if opts.diagR
  R=diag(diag(R));
end
if opts.sphericalR
    %R1=robCov(z,95);
    %R=R1+R2;
    nR=size(R,1);
    R=eye(nR)*trace(R)/nR;
end
R=R+1e-11*eye(size(R)); %Avoid numerical issues from PSD, but not PD, matrices

%Estimate x0,P0:
if isa(X,'cell')
    [x0,P0]=cellfun(@(x,p) estimateInit(x,p,A,Q),X,P,'UniformOutput',false);
else
    [x0,P0]=estimateInit(X,P,A,Q);
end

[A,B,C,x0,~,Q,P0] = canonizev2(A,B,C,x0,Q,P0);
end

function [x0,P0]=estimateInit(X,P,A,Q)
x0=X(:,1); %Smoothed estimate
P0=P(:,:,1); %Smoothed estimate, the problem with this estimate is that it is monotonically decreasing on the iteration of trueEM(). More likely it should converge to the same prior uncertainty we have for all other states.
%A variant to not make it monotonically decreasing:
%aux=mycholcov(P0);
%Aa=A*aux';
%P0=Q+Aa*Aa';
P0=Q;
end

function [yx,yu,xx,uu,xu,SP,SPt,xx_,uu_,xu_,xx1,xu1,SP_,S_P]=computeRelevantMatrices(Y,X,U,P,Pt,robustFlag)
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
%    if ~robustFlag
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
%     else
%         fun=@(x,y) squeeze(median(x,y));
%         N=size(X,2);
%         U2=reshape(U',[1,N,size(U,1)]);
%         X2=reshape(X',[1,N,size(X,1)]);
%         %Data for A,B estimation:
%         xu_=(N-1)*fun(X(:,1:end-1).*U2(1,1:end-1,:),2);
%         uu_=(N-1)*fun(U(:,1:end-1).*U2(1,1:end-1),2);
%         xx_=(N-1)*fun(X(:,1:end-1).*X2(1,1:end-1,:),2);
%         SP_=(N-1)*fun(P(:,:,1:end-1),3); %=SP-P(:,:,end);
%         S_P=(N-1)*fun(P(:,:,2:end),3); %=SP-P(:,:,1);
%         SPt=(N-1)*fun(Pt,3);
%         xu1=(N-1)*fun(X(:,2:end).*U2(1,1:end-1,:),2);
%         xx1=(N-1)*fun(X(:,2:end).*X2(1,1:end-1,:),2);
%         %Remove data associated to NaN values:
%         if any(any(isnan(Y)))
%           idx=~any(isnan(Y));
%           Y=Y(:,idx);
%           X=X(:,idx);
%           U=U(:,idx);
%           P=P(:,:,idx);
%         end
%         N=size(X,2);
%         U2=reshape(U',[1,N,size(U,1)]);
%         X2=reshape(X',[1,N,size(X,1)]);
%         %Data for C,D estimation:
%         SP=N*fun(P,3);
%         xu=N*fun(X.*U2,2);
%         uu=N*fun(U.*U2,2);
%         xx=N*fun(X.*X2,2);
%         yu=N*fun(Y.*U2,2);
%         yx=N*fun(Y.*X2,2);
%     end
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
