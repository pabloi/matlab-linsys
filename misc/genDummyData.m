N=1e3;
D=*2;
Xactual=ones(D,N);
Dy=D/2;
Y=nan(Dy,N);
A=[1 Ts; 0 1]; %Basic dynamic model
Atrue=zeros(D,D);
for i=1:D
  Atrue(i+[0,D],i+[0,D])=A;
end
Ctrue=[eye(Dy) zeros(D)];
Qtrue=5*eye(Dy);
Rtrue=eye(Dy)+.4;
for i=2:N
Xactual(:,i)=Atrue*Xactual(:,i-1) + Qtrue*randn(D,1);
Y(:,i)=Ctrue*Xactual(:,i) + Rtrue*randn(Dy,1);
end