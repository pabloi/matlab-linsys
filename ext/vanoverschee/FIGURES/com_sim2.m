% 
%   Reference:
%   
%           Subspace Identification for Linear Systems
%           Theory - Implementation - Applications
%           Peter Van Overschee / Bart De Moor
%           Kluwer Academic Publishers, 1996
%           Generation of Figure 4.9 on Page 132
%           
%   Copyright:
%   
%           Peter Van Overschee, December 1995
%           peter.vanoverschee@esat.kuleuven.ac.be
%

% 

j = 5000; 				% # columns
mini = 2; 				% # Minimal i
maxi = 10; 				% Maximal i
Ns = 100; 				% Number of experiments

% Take a simple first order system
a = 0.949;b = 1.8805;c = 0.8725;d = -2.0895;
k = -0.0580;r = 6.705;

% set up everything
ev1=[];ev2=[];ev3=[];
ze1=[];ze2=[];ze3=[];
pr1=[];pr2=[];pr3=[];

[bf,af]=butter(2,0.05); 		% Butterworth /  input
ue=randn(j+2*maxi,1); 			% Input
ue=dlsim(bf,af,ue); 			% Filtered input
ue=ue+randn(size(ue))*0.1; 		% Extra noise
u=ue; 					% Final input
yd=dlsim(a,b,c,d,u); 			% Deterministic output

m=1;l=1;
for i=mini:maxi
  
  clc
  disp(' ')
  disp(['   Number of block rows equal to: ',num2str(i)]) % Display i
  disp(' ')
  disp(' ')

  evs1=[];evs2=[];evs3=[];
  zes1=[];zes2=[];zes3=[];

  for N=1:Ns
    disp(['      Experiment number: ',num2str(N)]);
    en = randn(j+2*i,1)*sqrt(r); 		% Noise sequence
    ys = dlsim(a,k,c,1,en); 		% Stochastic output
    y = yd(1:length(ys))+ys;		% Total output
    u = ue(1:length(ys));
    
    [a1,b1,c1,d1,k1,r1,AUX] = com_alt(y,u,i,1,[],[],1);
    [a2,b2,c2,d2,k2,r2] = com_stat(y,u,i,1,AUX,[],1);
    [a3,b3,c3,d3,k3,r3] = subid(y,u,i,1,AUX,[],1);

    evs1(N)=a1;
    zes1(N)=a1-b1*c1/d1;
    evs2(N)=a2;
    zes2(N)=a2-b2*c2/d2;
    evs3(N)=a3;
    zes3(N)=a3-b3*c3/d3;
  end

  ev1(i)=mean(evs1);
  ze1(i)=mean(zes1);
  ev2(i)=mean(evs2);
  ze2(i)=mean(zes2);
  ev3(i)=mean(evs3);
  ze3(i)=mean(zes3);
	
end

ax=[mini:maxi];

hold off;subplot
clg
subplot(331)
plot(ax,ev1(ax),'*',[1,maxi],[a,a])
ax1=axis;
subplot(332)
plot(ax,ev2(ax),'*',[1,maxi],[a,a])
ax2=axis;
subplot(333)
plot(ax,ev3(ax),'*',[1,maxi],[a,a])
ax3=axis;
axn=[1,maxi,0.8,1];
subplot(331);axis(axn);
subplot(332);axis(axn);
subplot(333);axis(axn);

rez=a-b*c/d;
subplot(334)
plot(ax,ze1(ax),'*',[1,maxi],[rez,rez])
ax1=axis;
subplot(335)
plot(ax,ze2(ax),'*',[1,maxi],[rez,rez])
ax2=axis;
subplot(336)
plot(ax,ze3(ax),'*',[1,maxi],[rez,rez])
ax3=axis;
axn=[1,maxi,min(ax1(3),min(ax2(3),ax3(3))),max(ax1(4),max(ax2(4),ax3(4)))];
subplot(334);axis(axn);
subplot(335);axis(axn);
subplot(336);axis(axn);

subplot(331);title('Algorithm 1')
ylabel('Pole : A')
subplot(332);title('Algorithm 2')
subplot(333);title('Algorithm 3')
subplot(334);ylabel('Zero : A - BC/D')
subplot(334);xlabel('Number of block rows i')
subplot(335);xlabel('Number of block rows i')
subplot(336);xlabel('Number of block rows i')


