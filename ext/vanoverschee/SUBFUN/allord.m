% 
% [ersa,erpa,AUX] = allord(y,u,i,nall,AUX,W)
% 
%     Does subspace identification subid for all orders in the
%     vector nall and plots the simulation (ersa) and prediction (ersp) 
%     errors.
%     
% Copyright: 
%          Peter Van Overschee, December 1995
%          peter.vanoverschee@esat.kuleuven.ac.be
%


function [ersa,erpa,AUX] = allord(y,u,i,nall,AUX,W)

if (nargin < 6);W = [];end
if (nargin < 5);AUX = [];end

ersa = [];
erpa = [];
for k = 1:length(nall)
  n = nall(k);
  if (k == 1);
    disp('      Starting up ...')
  end
  disp(['        Identifying order: ',int2str(n)]);
 
  % Run the algorithms silently
  [A,B,C,D,K,R,AUX] = subid(y,u,i,n,AUX,W,1);
  if (u ~= []);
    [ys,ers] = simul(y,u,A,B,C,D);
    ersa(n,:) = ers;
  end
  [yp,erp] = predic(y,u,A,B,C,D,K);
  erpa(n,:) = erp;
end

tt = gcf;
ax = [1:max(nall)];
if (u ~= [])
  figure(tt);
  hold off;subplot
  bar(ax,ersa(ax,:));
  axis([0,max(nall)+1,0,min(max(max(ersa))+5,100)]);
  title('Simulation errors');xlabel('Order');
  figure(tt+1);
else
  figure(tt);
end
hold off;subplot
bar(ax,erpa(ax,:));axis([0,max(nall)+1,0,min(max(max(erpa))+5,100)]);
title('Prediction errors');xlabel('Order');

if (u ~= []);figure(tt);end 