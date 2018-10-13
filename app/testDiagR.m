load EMrealDimCompare1500v2.mat
model1=model;
load EMrealDimCompare1500_diagRv2.mat
model2=model;
load EMrealDimCompare1500_threshR.mat
model3=model;
load EMrealDimCompare1500_threshR_seeded.mat
model4=model;

%%
subModel=[model1(4) model2(4)  model1(4) model3(4) model4(4)];

%Cropping full R model:
subModel{3}.name='Cropped R, 3';
cc=subModel{3}.R;
dd=cc./sqrt(diag(cc));
dd=dd./sqrt(diag(cc))'; %Correlation matrix
cc=cc.*(abs(dd)>.23);
subModel{3}.R=cc;
min(eig(cc))

%Renaming thesholded R model:
subModel{4}.name='Thresholded R, 3';

%Renaming diagonal R model:
subModel{2}.name='Diagonal R, 3';
subModel{1}.name='EM, 3';
subModel{5}.name='Thresholded R, seeded';

%%
vizDataFit(subModel,Yf,Uf)
vizModels(subModel)
