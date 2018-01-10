function [E,M] = ePCA(Y)
%This function takes a matrix Y and finds a factorization of the form:
%Y=E*M; where the matrix E's columns are all exponentially decaying
%functions.
%The factorization is done by pre-setting the exponential functions to
%cover a range of decay rates log-uniformly distributed in the time
%interval given, and doing least-squares minimization.
%An additional loss term to encourage sparsity (L1 norm) can be added.

N=size(Y,1);
t=[0:N-1]';
stp=.1; 
minRate=5; %This acts as a regularization term, not allowing very fast exponentials, which would just fit the data of the first few samples and be 0 the rest of the time.
rates=[0 fliplr(1./exp([log(minRate):stp:log(N)]))];
rates=[0 1/35 1/500];
E=[exp(-t*rates) t/t(end)];

%Eliminate NaNs by linear interp
nanidx=any(isnan(Y),2);
Y=interp1(t(~nanidx),Y(~nanidx,:),t,'linear','extrap');
mu=mean(Y,1);
%Y=bsxfun(@minus,Y,mu);

%LS w/o sparsity term:
M1=pinv(E)*Y;

%With sparsity: doing for each muscle at a time:
% M=nan(size(M1));
% for i=1:size(Y,2)
% B=lasso(E,Y(:,i));
% M(:,i)=B(:,89);
% end

%Using lsqnonneg [this assumes all exponentials are going in the same
%direction]
M=nan(size(E,2),size(Y,2));
for i=1:size(Y,2)
    YY=Y(:,i);
    pp=polyfit(t,YY,1);
    if pp(1)>0 %Increasing
        YY=-YY;
    end    
    offset=min(YY);
    YY=YY-offset;
    MM=lsqnonneg(E,YY);
    MM(1)=MM(1)+offset;
    if pp(1)>0 %Increasing
        MM=-MM;
    end    
    M(:,i)=MM;
end

%The next step (TO DO) is to select the exponentials that have meaningful
%contributions to the data (energy?) & keep only those. Perhaps also allow
%for some adjustment of the rates so they don't necessarily have to be one
%of the pre-determined values set at the beginning




end

