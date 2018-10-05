%load EMrealDimCompare1500comm_2inALT %Should do it with noD
%model2alt=model;
load EMrealDimCompare1500comm_3in_noD.mat

order=2;
subModel=[model(1:3,order+1)];
subModel{1}.name='4th, 1in';
subModel{2}.name='4th, 2in';
subModel{3}.name='4th, 3in';
%modelA=model2alt{order+1,2};
%modelA.D=[modelA.D(:,1) zeros(size(modelA.D(:,1))) modelA.D(:,2)];
%modelA.B=[modelA.B(:,1) zeros(size(modelA.B(:,1))) modelA.B(:,2)];
%modelA.name='4th, 2in ALT';
%subModel=[subModel;{modelA}];

vizDataFit(subModel(:),Yf,Uf)
vizSingleModel(subModel{1},Yf,Uf(1,:))
vizSingleModel(subModel{2},Yf,Uf(1:2,:))
vizSingleModel(subModel{3},Yf,Uf)
%vizSingleModel(subModel{4},Yf,Uf)
