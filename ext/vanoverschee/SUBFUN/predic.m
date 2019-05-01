% 
% [yp,erp] = predic(y,u,A,B,C,D,K,ax)
% 
% Description:
%       Prediction with the state space model A,B,C,D,K (one step ahead)
%       
%            xp_{k+1} = A xp_k + B u_k + K (yp_k - C xp_k - D u_k)
%              yp_k   = C xp_k + D u_k
%              
%       The initial state xp_0 is estimated from the first N points
%       Where N is equal to 3 times the order of the system.
%       
%       ax:  (optional) the time axis over which erp is computed.
%            Defaults to all the time points.
%       erp: the percentual error per output as in Formula 6.5 page 192
%            These errors are clipped at 100 percent
%       
% For pure stochastic systems, use:
%  
% [yp,erp] = predic(y,[],A,[],C,[],K,ax)
%
% Copyright: 
%          Peter Van Overschee, December 1995
%          peter.vanoverschee@esat.kuleuven.ac.be
%
%

function [yp,erp] = predic(y,u,A,B,C,D,K,ax)

if (u == [])
  % Pure stochastic systems
  ds_flag = 2;
elseif (nargin == 7) | (nargin == 8)
  ds_flag = 1;
  if (A == []) | (B == []) | (C == []) | (D == [])
    yp = [];erp = [];
    return
  end
  if (K == []);
    % Switch to simulation error
    [yp,erp] = simul(y,u,A,B,C,D,ax);
    return
  end
else
  error('Not the right number of input arguments in predic');
end

% Check the arguments
[ny,l] = size(y);if l > ny;y = y';[ny,l] = size(y);end
if (l < 0);error('Need a non-empty output vector');end
if (ds_flag == 1)
  [nu,m] = size(u);if m > nu;u = u';[nu,m] = size(u);end
  if (m < 0);error('Need a non-empty input vector');end
  if (nu ~= ny);error('Number of data points different in input and output');end
end

[n,ac] = size(A);
[cr,cc] = size(C);
[kr,kc] = size(K);
if (ac ~= n) |  (cr ~= l) | (cc ~= n) | (kr ~= n) | (kc ~= l) 
  error('Incompatible state space system');
end
if (ds_flag == 1)
  [br,bc] = size(B);[dr,dc] = size(D);
  if (br ~= n) | (bc ~= m) | (dr ~= l) | (dc ~= m) 
    error('Incompatible state space system');
  end
end
if nargin < 8;ax=[1:ny];end 		% The axis for error computation

if (max(ax) > ny) | (min(ax) < 1);ax = [1:ny];end

nd = min(3*n,ny); 			% Number of data to compute x0

% Estimate the initial state

% Make gamma
gam = zeros(nd*l,n);bb = zeros(n,1);
for k=1:n
  b = bb;b(k) = 1;
  dd = dimpulse(A,b,C,zeros(l,1),1,nd+1);
  dd=dd(2:nd+1,:)';
  dd=dd(:);
  gam(:,k)=dd;
end

if (ds_flag == 1)
  % Make Hd
  Hd = zeros(l*nd,m*nd);
  for k=1:m
    hh=dimpulse(A,B,C,D,k,nd);
    inp_ind=[k:m:nd*m];
    for k1=1:l
      out_ind=[k1:l:nd*l];
      Hd(out_ind,inp_ind)=toeplitz(hh(1:nd,k1),[D(k1,k);zeros(nd-1,1)]);
    end
  end
  
  % And now solve the set of equations for x0
  U = u(1:nd,:)';U = U(:);
  Y = y(1:nd,:)';Y = Y(:);
  x0 = gam\(Y - Hd*U);
else
  Y = y(1:nd,:)';Y = Y(:);
  x0 = gam\Y;
end

% Now you're ready to predict  
Ap = A - K*C; 				% prediction A
Cp = C;
if (ds_flag == 1)
  Bp = [B - K*D, K];
  Dp = [D,zeros(l,l)];
  yp = dlsim(Ap,Bp,Cp,Dp,[u,y],x0);
else
  yp = dlsim(Ap,K,Cp,zeros(l,l),y,x0);
end

erv = (y-yp);erv = erv(ax,:); % Error over axis
erp = sqrt(sum(erv.^2)./sum(y.^2))*100;
idx = find((erp > 100) | isnan(erp));erp(idx) = ones(size(idx))*100;



