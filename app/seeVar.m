%% Load real data:
[Y,Yf,Ycom,Uf]=groupDataToMatrixForm(false, true);
Y=median(Y,3)';
Yf=median(Yf,3)';
figure;
bw=31;
aux=medfilt1(sqrt(mean(diff(Y,[],2).^2,1)),bw);
plot(aux/nanmedian(aux(1:150)),'DisplayName','Instant std, full');
hold on;
aux=medfilt1(sqrt(mean(diff(Yf,[],2).^2,1)),bw);
plot(aux/nanmedian(aux(1:150)),'DisplayName','Instant std,sym')

%%
%% Load real data:
[Y,Yf,Ycom,Uf]=groupDataToMatrixForm(true,true);
Y=median(Y,3)';
Yf=median(Yf,3)';
aux=medfilt1(sqrt(mean(diff(Y,[],2).^2,1)),bw);
plot(aux/nanmedian(aux(1:150)),'DisplayName','Instant std, full,sqrt');
hold on;
aux=medfilt1(sqrt(mean(diff(Yf,[],2).^2,1)),bw);
plot(aux/nanmedian(aux(1:150)),'DisplayName','Instant std,sym,sqrt')

[Y,Yf,Ycom,Uf]=groupDataToMatrixForm2(true,true);
Y=median(Y,3)';
Yf=median(Yf,3)';
aux=medfilt1(sqrt(mean(diff(Y,[],2).^2,1)),bw);
plot(aux/nanmedian(aux(1:150)),'DisplayName','Instant std, full,log');
hold on;
aux=medfilt1(sqrt(mean(diff(Yf,[],2).^2,1)),bw);
plot(aux/nanmedian(aux(1:150)),'DisplayName','Instant std,sym,log')
legend
grid on

%%
figure
i=2;
[Y,Yf,Ycom,Uf]=groupDataToMatrixForm(false, true);
histogram(Yf(i,:),'EdgeColor','none','FaceAlpha',.5,'DisplayName','raw')
hold on
[Y,Yf,Ycom,Uf]=groupDataToMatrixForm(true, true);
histogram(Yf(i,:),'EdgeColor','none','FaceAlpha',.5,'DisplayName','sqrt')
[Y,Yf,Ycom,Uf]=groupDataToMatrixForm2(true, true);
histogram(Yf(i,:),'EdgeColor','none','FaceAlpha',.5,'DisplayName','log')
legend
