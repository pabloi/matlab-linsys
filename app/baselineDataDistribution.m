[Y,Yf]=groupDataToMatrixForm(false,true);
figure; for i=1:180; subplot(3,5,floor((i-1)/12)+1); hold on; histogram(sqrt(Y(i,1:150)),'EdgeColor','none'); end
figure; for i=1:180; subplot(3,5,floor((i-1)/12)+1); hold on; histogram((Y(i,1:150)),'EdgeColor','none'); end
figure; for i=1:180; subplot(3,5,floor((i-1)/12)+1); hold on; qqplot(sqrt(Y(i,1:150))); end
figure; for i=1:180; subplot(3,5,floor((i-1)/12)+1); hold on; qqplot((Y(i,1:150))); end
