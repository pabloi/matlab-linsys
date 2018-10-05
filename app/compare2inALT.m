load EMrealDimCompare1500.mat
model1=model;
load EMrealDimCompare1500_2inALT.mat
model2=model;
load EMrealDimCompare1500_2inALTnoD.mat
model3=model;

model1{5}.name='4th, 1in';
model2{5}.name='4th, 2in';
model3{5}.name='4th, 2in, noD';
subModel=[model1(5),model2(5),model3(5)];

vizDataFit(subModel,Yf,Uf)
vizSingleModel(model1{5},Yf,Uf(1,:))
vizSingleModel(model2{5},Yf,Uf)
vizSingleModel(model3{5},Yf,Uf)
