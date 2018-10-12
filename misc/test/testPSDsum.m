d1=200;
d2=3;
A1=randn(d1,d1-1);
C=10*A1(:,1:d2);
tol=1e-11;
A=A1*A1'+tol*eye(d1); %The smallest eigenvalue of A will be tol, same as for A+C*B*C'
[V,D]=eig(A);
orthDirection=V(:,1);
orthDirection=orthDirection/sum(orthDirection.^2); %x'*A*x =1e-11 for this vector
cA=mycholcov(A);
B=rand(d2);
B=B*B';


%
Niter=1e5;
sB=mycholcov(B);
  CsB=C*sB';
e1=0;
e2=0;
e3=0;
cA=mycholcov(A);
cP=myPSDsum(cA,CsB*CsB');
sum1=(A+CsB*CsB');
sum3=A+C*B*C';
sum2=cP'*cP;

%Checking eig: this has too many numerical errors, regardless of how well the matrix was computed:
%For all three matrices, sometimes the 'sum' is not even positive semidefinite!
%min(eig(sum1))
%min(eig(sum2))
%min(eig(sum3))

%Checking random projections: onto 'tol' subspace
x1=I*rand(d1,Niter);
x1=x1-A1*(A1\x1); %Orthogonal to A
x1=x1./sum(x1.^2,1);
cS1=mycholcov(sum1);
for i=1:Niter
  x=x1(:,i);
  x=orthDirection+1e-13*randn(d1,1);
  xC=x'*CsB;
  xA=x'*A1;
  y=xA*xA' + xC*xC' + tol*norm(x)^2;%D(1); %x'*(A+C*B*C')*x, but in the most accurate form I can think of, since I know A1, C and x exactly
  %y should be equal to tol, plus minus some noise from the projection and normalization
  %y=D(1);
  xcS1=x'*cS1';
  y1=xcS1*xcS1';
  xcP=x'*cP';
  y2=xcP*xcP';
  y3=x'*sum3*x;
  %e1=e1+(y1<0);
  %e2=e2+(y2<0);
  %e3=e3+(y3<0);
  e1=e1+(1-y1/y)^2 ;
  e2=e2+(1-y2/y)^2 ;
  e3=e3+(1-y3/y)^2 ;
end
disp(' "true" value was determined as y= (x''*a'')*(a*x) + (x''*Cb)*(Cb''*x), wher a=chol(A) and Cb=C*chol(B)')
disp('-----Relative error in estimation of x''*C*x for (normalized) x closely aligned to a dimension where x''*A*x ~0 and x''*B*x=0----')
disp(['Using (x''*c'')*(c*x) where c=chol(A+Cb*Cb'') and Cb=C*chol(B): ' num2str(e1/Niter)])
disp(['Using successive cholesky updates: ' num2str(e2/Niter)]) %I'd expect this to be the most precise, since it is the only one that guarantees PSD
%It is! by an order of magnitude in squared error, and NEVER returns a negative result
disp(['Using simply x''*(A+C*B*C'')*x: ' num2str(e3/Niter)])



%Checking random projections:
e1=0; e2=0; e3=0;
for i=1:Niter
  x=rand(d1,1);
  x=x/sum(x.^2);
  xC=x'*CsB;
  xA=x'*A1;
  y=xA*xA' + xC*xC' + tol*norm(x)^2;%D(1); %x'*(A+C*B*C')*x, but in the most accurate form I can think of, since I know A1, C and x exactly
  y1=x'*sum1*x;
  xcP=x'*cP';
  y2=xcP*xcP';
  y3=x'*sum3*x;
  e1=e1+(y-y1)^2 / y^2;
  e2=e2+(y-y2)^2 / y^2;
  e3=e3+(y-y3)^2 / y^2;
end
display('-------Relative error in estimation of x''*C*x for randomly chosen (normalized) x--------')
disp(['Using (x''*c'')*(c*x) where c=chol(A+Cb*Cb'') and Cb=C*chol(B): ' num2str(e1/Niter)])
disp(['Using successive cholesky updates: ' num2str(e2/Niter)]) %I'd expect this to be the most precise, since it is the only one that guarantees PSD
%It is! by an order of magnitude in squared error, and NEVER returns a negative result
disp(['Using simply x''*(A+C*B*C'')*x: ' num2str(e3/Niter)])
