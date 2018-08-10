%% Testing robCov() for unbiasedness under multinormal assumption
clearvars
M=6;
Qsqrt=10*randn(M,M);
Q=Qsqrt*Qsqrt';

%% Estimate:
Nreps=1e2;
Qest=nan(M,M,Nreps);
Qtrue=nan(M,M,Nreps);
for i=1:Nreps
% Generate data:
X=Qsqrt*randn(size(Q,2),1e3);
X=X+(rand(size(X))>.99)*1e2; %Add 1% outliers, to observe the robustness

% Estimate:
Qest(:,:,i)=robCov(X); %My robust estimate
Qtrue(:,:,i)=X*X'/size(X,2); %Standard, MLE, estimate given a known mean
end

%% Visualize
Q
norm(Q,'fro')
norm(Q-mean(Qtrue,3),'fro')
norm(Q-mean(Qest,3),'fro')