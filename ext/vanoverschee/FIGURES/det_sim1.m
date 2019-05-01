% 
%   Reference:
%   
%           Subspace Identification for Linear Systems
%           Theory - Implementation - Applications
%           Peter Van Overschee / Bart De Moor
%           Kluwer Academic Publishers, 1996
%           Generation of Figure 2.6 on Page 49
%           
%   Copyright:
%   
%           Peter Van Overschee, December 1995
%           peter.vanoverschee@esat.kuleuven.ac.be
%


a = 0.85;b = 0.5;c = -0.3;d = 0.1; 	% The system
Ns = 200; 				% Monte Carlo experiments
[bf,af] = butter(2,0.3); 		% Input filter
u = filter(bf,af,randn(1000,1)); 	% Input
ur = u + 0.1*randn(1000,1); 		% More input
i = 4; 					% Block rows
yr = dlsim(a,b,c,d,ur); 		% Simulated output
bc = [0.02,-0.041];ac = [1,-0.85]; 	% Coloring filter

disp(['   Total number of experiments: ',num2str(Ns)]);
% Start Monte Carlo experiments
for k=1:Ns
  disp(['      Experiment number: ',num2str(k)]);
  % White noise on outputs
  y = yr + 0.1*randn(1000,1);
  u = ur;
  
  [A13,du,du,du,AUX] = det_alt(y,u,i,1,[],1); % Theorem
  A11 = intersec(y,u,i,1,AUX,1); 	% Intersection
  A12 = project(y,u,i,1,AUX,1); 	% Projection
  a11(k) = A11; a12(k) = A12;a13(k) = A13; % Store
  
  
  % White noise on inputs and outputs
  y = yr + 0.1*randn(1000,1);
  u = ur + 0.1*randn(1000,1);
  
  [A23,du,du,du,AUX] = det_alt(y,u,i,1,[],1); % Theorem
  A21 = intersec(y,u,i,1,AUX,1); 	% Intersection
  A22 = project(y,u,i,1,AUX,1); 	% Projection
  a21(k) = A21; a22(k) = A22;a23(k) = A23; % Store
  
  % Colored noise on outputs
  y = yr + filter(bc,ac,randn(1000,1));
  u = ur;

  [A33,du,du,du,AUX] = det_alt(y,u,i,1,[],1); % Theorem
  A31 = intersec(y,u,i,1,AUX,1); 	% Intersection
  A32 = project(y,u,i,1,AUX,1); 	% Projection
  a31(k) = A31; a32(k) = A32;a33(k) = A33; % Store
end

  
subplot
subplot(331)
plot([1:Ns],a11,':',[1,Ns],[a,a],'-',[1,Ns],[mean(a11),mean(a11)],'-.')
title('Intersection')
ylabel('White on y_k')
ax1=axis;
subplot(332)
plot([1:Ns],a12,':',[1,Ns],[a,a],'-',[1,Ns],[mean(a12),mean(a12)],'-.')
title('Projection')
ax2=axis;
subplot(333)
plot([1:Ns],a13,':',[1,Ns],[a,a],'-',[1,Ns],[mean(a13),mean(a13)],'-.')
title('Theorem 2')
ax3=axis;
ax=[0,Ns,min(ax1(3),min(ax2(3),ax3(3))),max(ax1(4),max(ax2(4),ax3(4)))];
subplot(331);axis(ax);
subplot(332);axis(ax);
subplot(333);axis(ax);
subplot(334)


plot([1:Ns],a21,':',[1,Ns],[a,a],'-',[1,Ns],[mean(a21),mean(a21)],'-.')
ylabel('White on u_k, y_k')
ax1=axis;
subplot(335)
plot([1:Ns],a22,':',[1,Ns],[a,a],'-',[1,Ns],[mean(a22),mean(a22)],'-.')
ax2=axis;
subplot(336)
plot([1:Ns],a23,':',[1,Ns],[a,a],'-',[1,Ns],[mean(a23),mean(a23)],'-.')
ax3=axis;
ax=[0,Ns,min(ax1(3),min(ax2(3),ax3(3))),max(ax1(4),max(ax2(4),ax3(4)))];
subplot(334);axis(ax);
subplot(335);axis(ax);
subplot(336);axis(ax);
subplot(337)
plot([1:Ns],a31,':',[1,Ns],[a,a],'-',[1,Ns],[mean(a31),mean(a31)],'-.')
ylabel('Colored on y_k')
ax1=axis;
subplot(338)
plot([1:Ns],a32,':',[1,Ns],[a,a],'-',[1,Ns],[mean(a32),mean(a32)],'-.')
ax2=axis;
subplot(339)
plot([1:Ns],a33,':',[1,Ns],[a,a],'-',[1,Ns],[mean(a33),mean(a33)],'-.')
ax3=axis;
ax=[0,Ns,min(ax1(3),min(ax2(3),ax3(3))),max(ax1(4),max(ax2(4),ax3(4)))];
subplot(337);axis(ax);xlabel('Experiment')
subplot(338);axis(ax);xlabel('Experiment')
subplot(339);axis(ax);xlabel('Experiment')

