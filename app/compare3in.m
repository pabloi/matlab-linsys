load EMrealDimCompare1500_3in_noD.mat

subModel=[model(1:3)];

vizDataFit(subModel,Yf,Uf)
vizSingleModel(model{1},Yf,Uf(1,:))
vizSingleModel(model{2},Yf,Uf(1:2,:))
vizSingleModel(model{3},Yf,Uf)
