function [a,b,c,d,ko,x0]=x2thf(par,T,aux)
n=aux(1);
m=aux(2);
l=aux(3);
par=par';
if l < n
  a=[zeros(n-l,l),eye(n-l)];
  for k=1:l
    a=[a;par((k-1)*n+1:k*n)'];
  end
  ba = l*n;
else
  a = [];
  for k=1:n
    a = [a;par((k-1)*n+1:k*n)'];
  end
  ba=n*n;
end

b=[];
for k=1:m
  b=[b,par(ba+(k-1)*n+1:ba+k*n)];
end
ba = ba + n*m;

if l < n
  c = [eye(l),zeros(l,n-l)];
else
  c = eye(n);
  for k = 1:l-n
   c = [c;par(ba+(k-1)*n+1:ba+k*n)']; 
 end
 ba = ba + (l-n)*n;
end

d=[];
for k=1:m
  d=[d,par(ba+(k-1)*l+1:ba+k*l)];
end    
      
ko=[];ba=ba+l*m;
if (ba ~= length(par))
  for k=1:l
    ko=[ko,par(ba+(k-1)*n+1:ba+k*n)];
  end
else
  ko = zeros(n,l);
end

x0=zeros(n,1);


