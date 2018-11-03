
%% Get data:
squareFlag=false;
[Y,U,A,B,C,D,Q,R,x0,Yoff,Ysquared,Yexp,Ylap]=generateSyntheticData();
J=A;
B=[B zeros(size(B))];
D=[D, Yoff];
trueModel=autodeal(J,B,C,D,Q,R,x0,Y,U);
trueModel.name='True';
%%
U=[U;ones(size(U))];

% models:
opts.robustFlag=false;
opts.outlierReject=false;
opts.fastFlag=1;
opts.logFlag=true;
opts.indD=1:2;
opts.indB=1;
opts.Niter=1000;
for D1=1:3
    for k=1:4
      switch k
      case 1
        Yaux=Y;
      case 2
        Yaux=Ysquared;
      case 3
        Yaux=Yexp;
      case 4
        Yaux=Ylap;
      end
      tic
        if D1==0
          % Flat model:
          [J,B,C,D,Q,R]=getFlatModel(Yaux,U);
          model{1,k}=autodeal(J,B,C,D,Q,R);
          model{1,k}.name=['Flat'];
        else
        [fAh,fBh,fCh,D,fQh,R,fXh,fPh,logL,outLog]=randomStartEM(Yaux,U,D1,20,opts); %Slow/true EM
        model{D1+1,k}.runtime=toc;
        [J,B,C,X,~,Q,P] = canonize(fAh,fBh,fCh,fXh,fQh,fPh);
        model{D1+1,k}=autodeal(J,B,C,D,X,Q,R,P,logL,outLog);
        model{D1+1,k}.name=[num2str(D1) ' states']; %Robust mode does not do fast filtering
      end
    end
end
%%
save ./EM/test/EMsynth_squaredNoisy.mat

vizDataFit(model(2:4,1),Y,U)
vizDataFit(model(2:4,2),Ysquared,U)
vizDataFit(model(2:4,3),Yexp,U)
vizDataFit(model(2:4,4),Ylap,U)

vizModels([{trueModel}, model(3,:)]) %Correct order, different data
