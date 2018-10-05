load EMrealDimCompare1500.mat
model1=model;
load EMrealDimCompare1500_2in.mat
model2=model;

model1{5}.name='4th, 1in';
model2{5}.name='4th, 2in';
subModel=[model1(5),model2(5)];

vizDataFit(subModel,Yf,Uf)
vizSingleModel(model1{5},Yf,Uf(1,:))
vizSingleModel(model2{5},Yf,Uf)
