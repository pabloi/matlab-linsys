[Y,Ysym,Ycom,U]=groupDataToMatrixForm();

%
Yf=Ysym;

%Pre-process:
Yf=Yf-nanmedian(Yf(1:50,:,:)); %Subtracting Baseline
Yf=median(Yf,3); %Median across subjects

%Replace NaNs, for PCA:
addpath(genpath('../'))
Yf=substituteNaNs(Yf);
Yf=Yf(1:1350,:);
Yf=Yf';
U=U(:,1:1350);

%
figure;
subplot(1,2,1)
hold on
[pp,cc,aa]=pca(projectPerp(Yf,U)','Centered',false);
scatter(1:numel(aa),aa/sum(aa),50,'filled','DisplayName','Relative to total energy')
hold on
%[pp1,cc1,aa1]=pca(randn(size(Yf')),'Centered',true);
%scatter(1:numel(aa1),aa1/sum(aa1),50,'filled')
plot([1,numel(aa)],[1 1]/numel(aa),'k--')
set(gca,'YScale','log','XLim',[1 12],'YLim',aa([12,1])/sum(aa))
grid on
title('PCA eigenvalues')
scatter(1:numel(aa),[numel(aa):-1:1]'.*aa./(sum(aa)-cumsum(aa)),50,'filled','DisplayName','Relative to residual energy')
%scatter(1:numel(aa),[numel(aa):-1:1]'.*aa1./(sum(aa1)-cumsum(aa1)),50,'filled')
grid on
xlabel('Dimension')
ylabel('% variance')

%Subspace-style dimension analysis:
N=size(Y,2);
i=20;
j=N-2*i;
%Yf=Yf-mean(Yf,2);
Y_1i=myhankel(Yf,i,j);
U_1i=myhankel(U,i,j);
W_1i=[U_1i; Y_1i];
U_ip12i=myhankel(U(:,(i+1):end),i,j);
Y_ip12i=myhankel(Yf(:,(i+1):end),i,j);

O_ip1=(projectPerp(Y_ip12i,U_ip12i)/projectPerp(W_1i,U_ip12i))*W_1i;
[~,S,V] = svd(O_ip1,'econ');
sd=(diag(S));
subplot(1,2,2)
%plot(cumsum(sd)/sum(sd))
scatter(1:numel(sd),sd.^2/sum(sd.^2),50,'filled')
hold on
plot([1,numel(sd)],[1 1]/numel(sd),'k--')
plot([1,numel(sd)],[1 1]/size(Y,1),'k--')
set(gca,'YScale','log','XLim',[2,12],'YLim',sd([12,2]).^2/sum(sd.^2))
grid on
title('SVD from subspace method')
xlabel('Dimension')
ylabel('% variance')
