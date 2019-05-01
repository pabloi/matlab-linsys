% 
%   Reference:
%   
%           Subspace Identification for Linear Systems
%           Theory - Implementation - Applications
%           Peter Van Overschee / Bart De Moor
%           Kluwer Academic Publishers, 1996
%           Generation of Figure 3.14 on Page 92
%           
%   Copyright:
%   
%           Peter Van Overschee, December 1995
%           peter.vanoverschee@esat.kuleuven.ac.be
%


a = 0.949;k = 0.732;c = 0.1933;r = 0.8066; % The system
Ns = 100; 				% Monte Carlo experiments
mini = 2; 				% Minimal # Block rows 
maxi = 10; 				% Maximal # Block rows 
j = 1000; 				% Number of points
[g,l0] = kr2gl(a,k,c,r);

% Set up everything
ev1=[];ev2=[];ev3=[];
ze1=[];ze2=[];ze3=[];
pr1=[];pr2=[];pr3=[];
pra1=[];pra2=[];pra3=[];

for i=mini:maxi 			% Go over all i's
  
  clc
  disp(' ')
  disp(['   Number of block rows equal to: ',num2str(i)]) % Display i
  disp(' ')
  disp(' ')
    
  evs1=[];evs2=[];evs3=[];
  zes1=[];zes2=[];zes3=[];
  prs1=[];prs2=[];prs3=[];

  for N=1:Ns  % go over Monte Carlo experiments
    disp(['      Experiment number: ',num2str(N)]);
    e = randn(j+2*i,1)*sqrt(r); 	% A noise sequence
    y=dlsim(a,k,c,1,e);
    [a1,k1,c1,r1,AUX,g1,l01] = sto_stat(y,i,1,[],'UPC',1);
    fl1 = (k1 == []);
    [a2,k2,c2,r2,AUX,g2,l02] = sto_alt(y,i,1,AUX,'UPC',1);
    fl2 = (k2 == []);
    [a3,k3,c3,r3,AUX,g3,l03] = sto_pos(y,i,1,AUX,'UPC',1);
    fl3 = (k3 == []);
    		
    evs1(N)=a1;
    zes1(N)=a1-g1*c1/l01;
    prs1(N)=fl1;
    evs2(N)=a2;
    zes2(N)=a2-g2*c2/l02;
    prs2(N)=fl2;
    evs3(N)=a3;
    zes3(N)=a3-g3*c3/l03;
    prs3(N)=fl3;
	      
  end

  ev1(i)=mean(evs1);
  ze1(i)=mean(zes1);
  pr1(i)=100-sum(prs1)/Ns*100;
  ev2(i)=mean(evs2);
  ze2(i)=mean(zes2);
  pr2(i)=100-sum(prs2)/Ns*100;
  ev3(i)=mean(evs3);
  ze3(i)=mean(zes3);
  pr3(i)=100-sum(prs3)/Ns*100;
	
end

% Plot everything
ax=[mini:maxi];

subplot(331)
plot(ax,ev1(ax),'*',[1,maxi],[a,a])
ax1=axis;
subplot(332)
plot(ax,ev2(ax),'*',[1,maxi],[a,a])
ax2=axis;
subplot(333)
plot(ax,ev3(ax),'*',[1,maxi],[a,a])
ax3=axis;
%axn=[1,maxi,min(ax1(3),min(ax2(3),ax3(3))),max(ax1(4),max(ax2(4),ax3(4)))];
axn=[1,maxi,0.7,1];
subplot(331);axis(axn);
subplot(332);axis(axn);
subplot(333);axis(axn);

rez=a-g*c/l0;
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


subplot(337)
plot(ax,pr1(ax),'*',[1,maxi],[100,100])
ax1=axis;
subplot(338)
plot(ax,pr2(ax),'*',[1,maxi],[100,100])
ax2=axis;
subplot(339)
plot(ax,pr3(ax),'*',[1,maxi],[100,100])
ax3=axis;
axn=[1,maxi,min(ax1(3),min(ax2(3),ax3(3))),110];
subplot(337);axis(axn);
subplot(338);axis(axn);
subplot(339);axis(axn);

subplot(331);title('Algorithm 1')
ylabel('Pole : A')
subplot(332);title('Algorithm 2')
subplot(333);title('Algorithm 3')
subplot(334);ylabel('Zero : A - GC/L0')
subplot(337);xlabel('Number of block rows i')
subplot(338);xlabel('Number of block rows i')
subplot(339);xlabel('Number of block rows i')
subplot(337);ylabel('% posit. real')



