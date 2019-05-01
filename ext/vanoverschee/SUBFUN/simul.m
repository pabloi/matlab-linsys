% 
% [ys,ers] = simul(y,u,A,B,C,D,ax)
% 
% Description:
%       Simulation of a state space model A,B,C,D
%       
%            xs_{k+1} = A xs_k + B u_k
%              ys_k   = C xs_k + D u_k
%              
%       The initial state xs_0 is estimated from the first N points
%       Where N is equal to 3 times the order of the system.
%       
%       ax:  (optional) the time axis over which ers is computed.
%            Defaults to all the time points.
%       ers: the percentual error per output as in Formula 6.5 page 192
%            These errors are clipped at 100 percent
%
% Copyright: 
%          Peter Van Overschee, December 1995
%          peter.vanoverschee@esat.kuleuven.ac.be
%
%

function [ys,ers] = simul(y,u,A,B,C,D,ax)

if (nargin ~= 6) & (nargin ~= 7)
  error('Not the right number of input arguments in simul');
end

if (A == []) | (B == []) | (C == []) | (D == [])
  ys = [];ers = [];
  return
end

% Check the arguments
[ny,l] = size(y);if l > ny;y = y';[ny,l] = size(y);end
[nu,m] = size(u);if m > nu;u = u';[nu,m] = size(u);end
if (l < 0);error('Need a non-empty output vector');end
if (m < 0);error('Need a non-empty input vector');end
if (nu ~= ny);error('Number of data points different in input and output');end

[n,ac] = size(A);[br,bc] = size(B);
[cr,cc] = size(C);[dr,dc] = size(D);

if (ac ~= n) | (br ~= n) | (bc ~= m) | (cr ~= l) | (cc ~= n) | ...
      (dr ~= l) | (dc ~= m) 
  error('Incompatible state space system');
end

if nargin < 7;ax=[1:nu];end 		% The axis for error computation
if (max(ax) > nu) | (min(ax) < 1);ax = [1:nu];end

nd = min(3*n,nu); 			% Number of data to compute x0


% Solve x0 from: y = gamma . x0 + Hd . u (IO matrix equation)
 
% First make gamma
gam = zeros(nd*l,n);bb = zeros(n,1);
for k=1:n
  b = bb;b(k) = 1;
  dd = dimpulse(A,b,C,zeros(l,1),1,nd+1);
  dd=dd(2:nd+1,:)';
  dd=dd(:);
  gam(:,k)=dd;
end

% Now make Hd
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

ys = dlsim(A,B,C,D,u,x0); % Simulated output
erv = (y-ys);erv = erv(ax,:); % Error over axis
ers = sqrt(sum(erv.^2)./sum(y.^2))*100;
idx = find((ers > 100) | isnan(ers));ers(idx) = ones(size(idx))*100;
