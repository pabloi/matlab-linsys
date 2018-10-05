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
U=U(:,1:1350);
%
%[pp,cc,aa]=pca(Yf,'Centered',false);
%figure
%plot(aa/sum(aa))
%hold on
%set(gca,'YScale','log')
%grid on
%D=(U'\Yf);
%[pp,cc,aa]=pca(Yf-U'*D,'Centered',false);
%plot(aa/sum(aa))

%
figure;
hold on
for d=1:4
[J,B,C,D,X,Q,R]=subspaceID(Yf',U,d);
[J,B,C,X,~,Q]=canonizev4(J,B,C,X,Q);
R=diag(diag(R));
%subplot(2,1,1); plot(X')
res=Yf(1:size(X,2),:)'-C*X-D*U(:,1:size(X,2));
aux=sqrt(sum(res.^2));
p=plot(aux);
bar(size(res,2)+100*d,mean(aux),'FaceColor',p.Color,'BarWidth',100,'EdgeColor','none')
logLperSamplePerDim=dataLogLikelihood(Yf',U,J,B,C,D,Q,R,[],[],'approx')
end
model=autodeal(J,B,C,D,Q,R,X);
vizSingleModel(model,Yf',U)
