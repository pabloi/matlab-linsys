%% This script compares succintly the BIC, AIC and LRT criteria for determining model order

figure
subplot(2,1,1)
x=[1:100,100:10:1000];
N=1500;
plot(x,x,'LineWidth',2,'DisplayName','AIC')
hold on
plot(x,x*log(N)/2,'LineWidth',2,'DisplayName','BIC')
plot(x,.5*chi2inv(.95,x),'LineWidth',2,'DisplayName',['LRT, \alpha=.05'])
plot(x,.5*chi2inv(.99,x),'LineWidth',2,'DisplayName',['LRT, \alpha=.01'])
plot(x,.5*chi2inv(.999,x),'LineWidth',2,'DisplayName',['LRT, \alpha=.001'])
xlabel('Additional number of parameters')
ylabel('Necessary \Delta log-L')
title(['\Delta log-L vs. additional params. for ' num2str(N) ' samples'])
set(gca,'XScale','log','YScale','log')
grid on
legend('Location','Best')
subplot(2,1,2)
plot(x,1-chi2cdf(2*x,x),'LineWidth',2,'DisplayName','AIC')
hold on
plot(x,1-chi2cdf(x*log(N),x),'LineWidth',2,'DisplayName','BIC')
xlabel('Additional number of parameters')
ylabel('\alpha equivalent for LRT')
grid on
legend('Location','Best')
set(gca,'XScale','log','YScale','log','YLim',[1e-8 1])
title('BIC and AIC equivalent thresholds for LRT')