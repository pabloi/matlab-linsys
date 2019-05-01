% 
%   Reference:
%   
%           Subspace Identification for Linear Systems
%           Theory - Implementation - Applications
%           Peter Van Overschee / Bart De Moor
%           Kluwer Academic Publishers, 1996
%           Generation of Figure 4.5 on Page 116
%           
%   Copyright:
%   
%           Peter Van Overschee, December 1995
%           peter.vanoverschee@esat.kuleuven.ac.be
%

Nmax=4000; 				% Maximum j
Nstep=10; 				% j step
Nstart=100; 				% Start j
i=10; 					% #block rows

% The system
a = [0.6,0.6,0;-0.6,0.6,0;0,0,0.4];
b = [1.6161;-0.3481;2.6391];
k = [-1.1472;-1.5204;-3.1993];
c = [-0.4373,-0.5046,0.0936];
d = -0.7759;
r = 0.0432;

[bf,af]=butter(2,0.3);	% Filter for the input

en = randn(Nmax+2*i,1)*sqrt(r); 	% Noise
ys = dlsim(a,k,c,1,en); 		% Stochastic output
ue = randn(Nmax+2*i,1); 		% Start making the input
ue = dlsim(bf,af,ue);
ue = ue+randn(size(ue))*0.1;
u = ue; 				% This is the input
yd = dlsim(a,b,c,d,u); 			% Deterministic output
y = yd+ys; 				% Combined output

ax=[Nstart:Nstep:Nmax];
ne=max(size(ax));
R=[];
s1=[];s2=[];s3=[];

nn=0;
Nb=0;

for k=1:ne
  disp(['   Computing for j = ',num2str(ax(k)),' points'])  
  % To  save time, we compute R recursively
  if k == 1
    Ye = blkhank(y(1:Nmax),2*i,ax(1));
    Ue = blkhank(u(1:Nmax),2*i,ax(1));		
    Nn=ax(1);
  else
    Ye = blkhank(y(ax(k-1)+1:Nmax+2*i),2*i,ax(k)-ax(k-1));
    Ue = blkhank(u(ax(k-1)+1:Nmax+2*i),2*i,ax(k)-ax(k-1));
    Nn=ax(k);
  end
  R=[R*sqrt(Nb),[Ue;Ye]]/sqrt(Nn);
  R=triu(qr(R'))';
  R=R(1:4*i,1:4*i);
	
  % The oblique projection :
  G1=R(3*i+1:4*i,1:3*i)*pinv(R(1:3*i,1:3*i));
  L1=G1(:,1:i);
  L2=G1(:,i+1:2*i);
  L3=G1(:,2*i+1:3*i);

  tt1=(L1*R(1:i,1:2*i)+L3*R(2*i+1:3*i,1:2*i));
  tt2=L3*R(2*i+1:3*i,2*i+1:3*i);
  obl=[L1,L3];

  % N4SID
  [u1,ss1,v1]=svd(obl);
  s1(:,k)=diag(ss1);
	
  % MOESP
  Rt=R(i+1:2*i,1:2*i);
  pim=eye(2*i) - Rt'*inv(Rt*Rt')*Rt;
  obll=[tt1*pim,tt2];

  [u2,ss2,v2]=svd(obll);
  s2(:,k)=diag(ss2);

  % CVA algorithm :
  Ryfl=[R(3*i+1:4*i,1:2*i)*pim,R(3*i+1:4*i,2*i+1:4*i)];
  W1=inv(sqrtm(Ryfl*Ryfl'));
  [u3,ss3,v3]=svd(W1*obll);
  s3(:,k)=diag(ss3);

  Nb=Nn;
end


hold off
subplot
subplot(311)
plot(ax,s1','-')
axis([0,4000,0,3])
title('Singular Values N4SID algorithm') 
subplot(312)
plot(ax,s2','-')
axis([0,4000,0,2]);
title('Singular Values MOESP algorithm') 
subplot(313) 
plot(ax,acos(s3')/pi*180,'-')
axis([0,4000,0,90])
title('Principal Angles CVA algorithm') 
xlabel('Number of block rows j')
ylabel('degrees')

