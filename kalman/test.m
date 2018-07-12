%% Test function for Kalman filter
%% Load/gen data
D=*2;
Xactual=ones(D,N);
Dy=D/2;
Y=nan(Dy,N);
A=[1 Ts; 0 1];
Atrue=zeros(D,D);
for i=1:D
  Atrue(i+[0,D],i+[0,D])=A;
end
Ctrue=[eye(Dy) zeros(D)];
for i=2:1e3
Xactual(:,i)=Atrue*Xactual(:,i-1) + Qtrue*randn(D,1);
Y(:,i)=Ctrue*Xactual(:,i) + Rtrue*randn(Dy,1);
end

%From here on we assume the observations are stored in a matrix Y [DxN]

%% Filter 1:
[Dy,N]=size(Y);
D=2*Dy;
A=[1 Ts; 0 1];
A1=zeros(D,D);
for i=1:D
  A1(i+[0,D],i+[0,D])=A;
end

C1=[eye(Dy) zeros(D)];
[Q1,R1]=learnQR(Y,A1,C1);
[X,P,Xp,Pp]=filterStationary(Y,A1,C1,Q1,R1);

%% Filter 2:
%This filter imposes a 'prior'
%by adding fake output in the difference between markers

%This computes all differences between all possible pairs of
%marker/coordinates. In reality we only want to compute
%within-coordinate difference pairs
M=[];
for i=1:D
  aux=zeros(i-1,D);
  for j=i+1:D
    aux(j-i,[i,j])=[1 -1];
  end
  M=[M;aux];
end
