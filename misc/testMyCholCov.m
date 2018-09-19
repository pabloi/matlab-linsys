%% Case 1: full rank:
M=6;
Nreps=1e4;
Q=randn(M);
P=Q'*Q;
disp('chol')
tic; for i=1:Nreps; L0=chol(P); end; toc;
disp('cholcov')
tic; for i=1:Nreps; L1=cholcov(P); end; toc;
disp('mycholcov')
tic; for i=1:Nreps; L2=mycholcov(P); end; toc;
norm(P-L0'*L0,'fro')
norm(P-L1'*L1,'fro')
norm(P-L2'*L2,'fro')

%% Case 2: under rank
M=6;
Nreps=1e3;
Q=(randn(M-2,M)); %Rank M-2
P=Q'*Q;
disp('chol')
tic; for i=1:Nreps; try [L0]=chol(P); catch; end; end; toc; %Can't do w/o try
disp('cholcov')
tic; for i=1:Nreps; L1=cholcov(P); end; toc;
disp('mycholcov')
tic; for i=1:Nreps; L2=mycholcov(P); end; toc;
norm(P-L0'*L0,'fro')
norm(P-L1'*L1,'fro')
norm(P-L2'*L2,'fro')

%% Case 3: zero-rank
M=6;
Nreps=1e3;
P=zeros(M);
disp('chol')
tic; for i=1:Nreps; try L0=chol(P); catch; end; end; toc; %Can't do w/o try
disp('cholcov')
tic; for i=1:Nreps; L1=cholcov(P); end; toc;
disp('mycholcov')
tic; for i=1:Nreps; L2=mycholcov(P); end; toc;
norm(P-L0'*L0,'fro')
norm(P-L1'*L1,'fro')
norm(P-L2'*L2,'fro')