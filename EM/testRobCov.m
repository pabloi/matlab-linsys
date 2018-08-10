%% Testing robCov() for unbiasedness under multinormal assumption
%this script compares the performance of the robCov() estimator vs. the
%standard MLE estimator of covariance of a sample with known mean (assumed
%mean =0), for data drawn from a randomly chosen multinormal distribution,
%for different values of the parameters (dimension size, sample size,
%contamination with outlier samples).
%%
clearvars
fh=figure;
Nreps=1e1;
cut=90;
%% Test as function of matrix size
subplot(4,2,1)
clear ae at
Nsamp=1e3;
for M=1:10
Qsqrt=randn(M,M);
Q=Qsqrt*Qsqrt';
Nreps=1e2;
% Estimate:

Qest=nan(M,M,Nreps);
Qtrue=nan(M,M,Nreps);
for i=1:Nreps
% Generate data:
X=Qsqrt*randn(size(Q,2),Nsamp);

% Estimate:
Qest(:,:,i)=robCov(X,cut); %My robust estimate
Qtrue(:,:,i)=X*X'/size(X,2); %Standard, MLE, estimate given a known mean
end

% Visualize
at(M)=norm(Q-mean(Qtrue,3),'fro')/norm(Q,'fro');
ae(M)=norm(Q-mean(Qest,3),'fro')/norm(Q,'fro');
end
p1=scatter(1:length(at),at,'filled','DisplayName','MLE');
hold on
p2=scatter(1:length(ae),ae,'filled','DisplayName','Robust');
title(['Nreps=' num2str(Nreps) ', Nsamples=' num2str(Nsamp) ', no outliers, reject=' num2str(100-cut) '%'])
xlabel('Covariance size (dimension) (M)')
ylabel('|\hat{Q}-Q|_F / |Q|_F')
legend

%% Test as function of data size
subplot(4,2,2)
clear ae at ae2 ae3
M=6;
for j=1:5
Qsqrt=randn(M,M);
Q=Qsqrt*Qsqrt';
Nreps=1e2;
% Estimate:

Qest=nan(M,M,Nreps);
Qtrue=nan(M,M,Nreps);
for i=1:Nreps
% Generate data:
X=Qsqrt*randn(size(Q,2),10^j);

% Estimate:
Qest(:,:,i)=robCov(X,cut); %My robust estimate
Qtrue(:,:,i)=X*X'/size(X,2); %Standard, MLE, estimate given a known mean
end

% Visualize
at(j)=norm(Q-mean(Qtrue,3),'fro')/norm(Q,'fro');
ae(j)=norm(Q-mean(Qest,3),'fro')/norm(Q,'fro');
end
p1=scatter(1:length(at),at,'filled','DisplayName','MLE');
hold on
p2=scatter(1:length(ae),ae,'filled','DisplayName','Robust');
title([num2str(M) ' x ' num2str(M) ' matrix, Nreps=' num2str(Nreps) ', no outliers, reject=' num2str(100-cut) '%'])
xlabel('Log_{10}(sample size)')
ylabel('|\hat{Q}-Q|_F / |Q|_F')
legend
set(gca,'YScale','log')
%% Test with outliers: single axis normal noise
subplot(4,2,3)
clear ae at ae2 ae3
M=6;
Nsamp=1e3;
for r=1:11 %Percent outliers
Qsqrt=randn(M,M);
Q=Qsqrt*Qsqrt';
Nreps=1e2;
% Estimate:

Qest=nan(M,M,Nreps);
Qest2=nan(M,M,Nreps);
Qest3=nan(M,M,Nreps);
Qtrue=nan(M,M,Nreps);
for i=1:Nreps
% Generate data:
X=Qsqrt*randn(size(Q,2),Nsamp);
X=X+1e1*repmat((rand(1,Nsamp)>(1-r/100)).*(randn(1,Nsamp)),size(Q,2),1); 
%Add r% outliers, to observe the robustness. Outliers are drawn from
%an univariate normal with var=100, mean=0, and aligned along the x1=x2=...=xM
%direction

% Estimate:
Qest(:,:,i)=robCov(X,cut); %My robust estimate
Qest2(:,:,i)=robCov(X,100-2*(100-cut)); %My robust estimate
Qest3(:,:,i)=robCov(X,100-.5*(100-cut)); %My robust estimate
Qtrue(:,:,i)=X*X'/size(X,2); %Standard, MLE, estimate given a known mean
end

% Visualize
at(r)=norm(Q-mean(Qtrue,3),'fro')/norm(Q,'fro');
ae(r)=norm(Q-mean(Qest,3),'fro')/norm(Q,'fro');
ae2(r)=norm(Q-mean(Qest2,3),'fro')/norm(Q,'fro');
ae3(r)=norm(Q-mean(Qest3,3),'fro')/norm(Q,'fro');
end
p1=scatter(1:length(at),at,'filled','DisplayName','MLE');
hold on
p3=scatter(1:length(ae),ae2,'filled','DisplayName',['Robust, reject=' num2str(2*(100-cut)) '%']);
p2=scatter(1:length(ae),ae,'filled','DisplayName',['Robust, reject=' num2str(100-cut) '%']);
p4=scatter(1:length(ae),ae3,'filled','DisplayName',['Robust, reject=' num2str(.5*(100-cut)) '%']);
title([num2str(M) ' x ' num2str(M) ' matrix, Nreps=' num2str(Nreps) ', Nsamples=' num2str(Nsamp) ', Single axis normal noise'])
xlabel('% outliers')
ylabel('|\hat{Q}-Q|_F / |Q|_F')
legend
set(gca,'YScale','log')
%% Test with outliers: ring noise
subplot(4,2,4)
clear ae at ae2 ae3
M=6;
Nsamp=1e3;
for r=1:11 %Percent outliers
Qsqrt=randn(M,M);
Q=Qsqrt*Qsqrt';
Nreps=1e2;
% Estimate:

Qest=nan(M,M,Nreps);
Qest2=nan(M,M,Nreps);
Qest3=nan(M,M,Nreps);
Qtrue=nan(M,M,Nreps);
for i=1:Nreps
% Generate data:
X=Qsqrt*randn(size(Q,2),Nsamp);
aux=randn(size(X));
aux=1e3*aux/sqrt(sum(aux.^2,1));
X=X+(rand(1,Nsamp)>(1-r/100)).*aux; 
%Add r% outliers

% Estimate:
Qest(:,:,i)=robCov(X,cut); %My robust estimate
Qest2(:,:,i)=robCov(X,100-2*(100-cut)); %My robust estimate
Qest3(:,:,i)=robCov(X,100-.5*(100-cut)); %My robust estimate
Qtrue(:,:,i)=X*X'/size(X,2); %Standard, MLE, estimate given a known mean
end

% Visualize
at(r)=norm(Q-mean(Qtrue,3),'fro')/norm(Q,'fro');
ae(r)=norm(Q-mean(Qest,3),'fro')/norm(Q,'fro');
ae2(r)=norm(Q-mean(Qest2,3),'fro')/norm(Q,'fro');
ae3(r)=norm(Q-mean(Qest3,3),'fro')/norm(Q,'fro');
end
p1=scatter(1:length(at),at,'filled','DisplayName','MLE');
hold on
p3=scatter(1:length(ae),ae2,'filled','DisplayName',['Robust, reject=' num2str(2*(100-cut)) '%']);
p2=scatter(1:length(ae),ae,'filled','DisplayName',['Robust, reject=' num2str(100-cut) '%']);
p4=scatter(1:length(ae),ae3,'filled','DisplayName',['Robust, reject=' num2str(.5*(100-cut)) '%']);
title([num2str(M) ' x ' num2str(M) ' matrix, Nreps=' num2str(Nreps) ', Nsamples=' num2str(Nsamp) ', ring noise'])
xlabel('% outliers')
ylabel('|\hat{Q}-Q|_F / |Q|_F')
legend
set(gca,'YScale','log')
%% Test with outliers: cluster noise
subplot(4,2,6)
clear ae at ae2 ae3
M=6;
Nsamp=1e3;
for r=1:11 %Percent outliers
Qsqrt=randn(M,M);
Q=Qsqrt*Qsqrt';
Nreps=1e2;
% Estimate:

Qest=nan(M,M,Nreps);
Qest2=nan(M,M,Nreps);
Qest3=nan(M,M,Nreps);
Qtrue=nan(M,M,Nreps);
for i=1:Nreps
% Generate data:
X=Qsqrt*randn(size(Q,2),Nsamp);
aux=randn(size(X))+10;
X=X+(rand(1,Nsamp)>(1-r/100)).*aux; 
%Add r% outliers

% Estimate:
Qest(:,:,i)=robCov(X,cut); %My robust estimate
Qest2(:,:,i)=robCov(X,100-2*(100-cut)); %My robust estimate
Qest3(:,:,i)=robCov(X,100-.5*(100-cut)); %My robust estimate
Qtrue(:,:,i)=X*X'/size(X,2); %Standard, MLE, estimate given a known mean
end

% Visualize
at(r)=norm(Q-mean(Qtrue,3),'fro')/norm(Q,'fro');
ae(r)=norm(Q-mean(Qest,3),'fro')/norm(Q,'fro');
ae2(r)=norm(Q-mean(Qest2,3),'fro')/norm(Q,'fro');
ae3(r)=norm(Q-mean(Qest3,3),'fro')/norm(Q,'fro');
end
p1=scatter(1:length(at),at,'filled','DisplayName','MLE');
hold on
p3=scatter(1:length(ae),ae2,'filled','DisplayName',['Robust, reject=' num2str(2*(100-cut)) '%']);
p2=scatter(1:length(ae),ae,'filled','DisplayName',['Robust, reject=' num2str(100-cut) '%']);
p4=scatter(1:length(ae),ae3,'filled','DisplayName',['Robust, reject=' num2str(.5*(100-cut)) '%']);
title([num2str(M) ' x ' num2str(M) ' matrix, Nreps=' num2str(Nreps) ', Nsamples=' num2str(Nsamp) ', cluster noise'])
xlabel('% outliers')
ylabel('|\hat{Q}-Q|_F / |Q|_F')
legend
set(gca,'YScale','log')
%% Test with outliers: different iteration number (to show Niter=2 is suff)
subplot(4,2,5)
clear ae at ae2 ae3

M=6;
Nsamp=1e3;
for r=1:11 %Percent outliers
Qsqrt=randn(M,M);
Q=Qsqrt*Qsqrt';
Nreps=1e2;
% Estimate:

Qest=nan(M,M,Nreps);
Qest2=nan(M,M,Nreps);
Qest3=nan(M,M,Nreps);
Qtrue=nan(M,M,Nreps);
for i=1:Nreps
% Generate data:
X=Qsqrt*randn(size(Q,2),Nsamp);
X=X+1e1*repmat((rand(1,Nsamp)>(1-r/100)).*(randn(1,Nsamp)),size(Q,2),1); 
%Add r% outliers
% Estimate:
Qest(:,:,i)=robCov(X,cut,1); %My robust estimate
Qest2(:,:,i)=robCov(X,cut,2); %My robust estimate
Qest3(:,:,i)=robCov(X,cut,10); %My robust estimate
Qtrue(:,:,i)=X*X'/size(X,2); %Standard, MLE, estimate given a known mean
end

% Visualize
at(r)=norm(Q-mean(Qtrue,3),'fro')/norm(Q,'fro');
ae(r)=norm(Q-mean(Qest,3),'fro')/norm(Q,'fro');
ae2(r)=norm(Q-mean(Qest2,3),'fro')/norm(Q,'fro');
ae3(r)=norm(Q-mean(Qest3,3),'fro')/norm(Q,'fro');
end
p1=scatter(1:length(at),at,'filled','DisplayName','MLE');
hold on
p3=scatter(1:length(ae),ae,'filled','DisplayName',['Robust, Niter=1']);
p2=scatter(1:length(ae),ae2,'filled','DisplayName',['Robust, Niter=2']);
p4=scatter(1:length(ae),ae3,'filled','DisplayName',['Robust, Niter=10']);
title([num2str(M) ' x ' num2str(M) ' matrix, Nreps=' num2str(Nreps) ', Nsamples=' num2str(Nsamp) ', reject=' num2str(100-cut) '%, single axis noise'])
xlabel('% outliers')
ylabel('|\hat{Q}-Q|_F / |Q|_F')
legend
set(gca,'YScale','log')

%% Compare performance and runtime to robustCov()

clear ae at ae2 ae3 rcTime rc2Time time
M=6;
Nsamp=1e3;
for r=1:5 %Percent outliers
Qsqrt=randn(M,M);
Q=Qsqrt*Qsqrt';
Nreps=1e2;
% Estimate:

Qest=nan(M,M,Nreps);
Qest2=nan(M,M,Nreps);
Qest3=nan(M,M,Nreps);
Qtrue=nan(M,M,Nreps);
for i=1:Nreps
% Generate data:
X=Qsqrt*randn(size(Q,2),Nsamp);
X=X+1e1*repmat((rand(1,Nsamp)>(1-2*r/100)).*(randn(1,Nsamp)),size(Q,2),1); 
%Add r% outliers
% Estimate:
tic
Qest(:,:,i)=robCov(X,cut); %My robust estimate
rcTime(i)=toc;
tic
Qest2(:,:,i)=robustcov(X'); %MAtlab built-in
rc2Time(i)=toc;
tic
Qtrue(:,:,i)=X*X'/size(X,2); %Standard, MLE, estimate given a known mean
time(i)=toc;
end

% Visualize
at(r)=norm(Q-mean(Qtrue,3),'fro')/norm(Q,'fro');
ae(r)=norm(Q-mean(Qest,3),'fro')/norm(Q,'fro');
ae2(r)=norm(Q-mean(Qest2,3),'fro')/norm(Q,'fro');
rt(r)=mean(time);
re(r)=mean(rcTime);
re2(r)=mean(rc2Time);
end
subplot(4,2,7)
p1=scatter([1:length(at)]*2,at,'filled','DisplayName','MLE');
hold on
p3=scatter([1:length(at)]*2,ae,'filled','DisplayName',['robCov()']);
p2=scatter([1:length(at)]*2,ae2,'filled','DisplayName',['robustcov()']);
title(['Performance comparison to robustcov() [same conds as above]'])
xlabel('% outliers')
ylabel('|\hat{Q}-Q|_F / |Q|_F')
legend
set(gca,'YScale','log')
subplot(4,2,8)
p1=scatter([1:length(at)]*2,rt,'filled','DisplayName','MLE');
hold on
p3=scatter([1:length(at)]*2,re,'filled','DisplayName',['RobCov()']);
p2=scatter([1:length(at)]*2,re2,'filled','DisplayName',['RobustCov()']);
xlabel('% outliers')
ylabel('Avg. run time (s)')
legend
set(gca,'YScale','log')
title('Run time comparison to robustcov()')
%% Save fig