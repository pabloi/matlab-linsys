%%
A=[.95,0; 0 , .999];
B=zeros(2,1); %Both states asymptote at 1
Ny=3; %Number of outputs
C=randn(Ny,2);
D=zeros(Ny,1);
Q=.1*eye(2);
R=.01*eye(Ny);
x0=[1;1];
U=[zeros(1,500), ones(1,1000)];
d=2;


%%
clear err1 err2 errn err1_ err2_ errn_
Niter=1e2;
for k=1:Niter
  %% Sim:
  [Y,X]=fwdSim(U,A,B,C,D,x0,Q,R);

  %%Estimates from noiseless data:
  A_LS1=X(:,2:end)/X(:,1:end-1);
  err1(:,k)=sort(eig(A_LS1))-sort(eig(A));
  A_LS2=sqrtm(X(:,3:end)/X(:,1:end-2));
  err2(:,k)=sort(eig(A_LS2))-sort(eig(A));
  A_LSn=((X(:,5:end)+X(:,4:end-1)+X(:,3:end-2)+X(:,2:end-3))/X(:,1:end-4));
  v=eig(A_LSn);
  clear w
  for l=1:size(X,1)
    w(l)=fzero(@(x) polyval([1 1 1 1 -v(l)],x),1);
  end
  errn(:,k)=sort(w')-sort(eig(A));

  %Estimates from noisy data:
  Xn=X+100*rand(size(X));
  A_LS1=Xn(:,2:end)/Xn(:,1:end-1);
  err1_(:,k)=sort(eig(A_LS1))-sort(eig(A));
  A_LS2=sqrtm(Xn(:,3:end)/Xn(:,1:end-2));
  err2_(:,k)=sort(eig(A_LS2))-sort(eig(A));
  A_LSn=((Xn(:,5:end)+Xn(:,4:end-1)+Xn(:,3:end-2)+Xn(:,2:end-3))/Xn(:,1:end-4));
  v=eig(A_LSn);
  clear w
  for l=1:size(X,1)
    w(l)=fzero(@(x) polyval([1 1 1 1 -v(l)],x),1);
  end
  errn_(:,k)=sort(w')-sort(eig(A));
end

%%
disp(['Noisless, A estim error: \mu=' num2str(mean(err1,2)',3) ' \pm  \sigma=' num2str(std(err1,[],2)',3) ]);
disp(['Noisless, A^2 estim error: \mu=' num2str(mean(err2,2)',3) ' \pm  \sigma=' num2str(std(err2,[],2)',3) ]);
disp(['Noisless, A+A^2+A^3+A^4 estim error: \mu=' num2str(mean(errn,2)',3) ' \pm  \sigma=' num2str(std(errn,[],2)',3) ]);

disp(['Noisy, A estim error: \mu=' num2str(mean(err1_,2)',3) ' \pm  \sigma=' num2str(std(err1_,[],2)',3) ]);
disp(['Noisy, A^2 estim error: \mu=' num2str(mean(err2_,2)',3) ' \pm  \sigma=' num2str(std(err2_,[],2)',3) ]);
disp(['Noisy, A+A^2+A^3+A^4  estim error: \mu=' num2str(mean(errn_,2)',3) ' \pm  \sigma=' num2str(std(errn_,[],2)',3) ]);
