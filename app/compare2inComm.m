load EMrealDimCompare1500comm_2inALT.mat


model{5,1}.name='4th, 1in';
model{5,2}.name='4th, 2in';
subModel=[model(5,1),model(5,2)];

vizDataFit(subModel,Yf,Uf)
vizSingleModel(model{5,1},Yf,Uf(1,:))
vizSingleModel(model{5,2},Yf,Uf)
