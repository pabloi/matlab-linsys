% 
%   Deterministic subspace identification (Intersection)
%
%           [A,B,C,D] = intersec(y,u,i);
% 
%   Inputs:
%           y: matrix of measured outputs
%           u: matrix of measured inputs
%           i: number of block rows in Hankel matrices
%              (i * #outputs) is the max. order that can be estimated 
%              Typically: i = 2 * (max order)/(#outputs)
%           
%   Outputs:
%           A,B,C,D: deterministic state space system
%           
%                  x_{k+1) = A x_k + B u_k        
%                    y_k   = C x_k + D u_k
%           
%   Optional:
%
%           [A,B,C,D,ss] = intersec(y,u,i,n,AUX,sil);
%   
%           n:    optional order estimate (default [])
%           AUX:  optional auxilary variable to increase speed (default [])
%           ss:   column vector with singular values
%           sil:  when equal to 1 no text output is generated
%           
%   Example:
%   
%           [A,B,C,D,AUX] = det_stat(y,u,10,2);
%           for k=3:6
%              [A,B,C,D] = intersec(y,u,10,k,AUX);
%           end
%           
%           
%   Note:
%           The variable AUX is not computed as an output by intersec.  
%           Variables AUX computed by det_stat or det_alt however can 
%           be used as inputs.
%
%   Reference:
%   
%           Subspace Identification for Linear Systems
%           Theory - Implementation - Applications
%           Peter Van Overschee / Bart De Moor
%           Kluwer Academic Publishers, 1996, Page 45
%
%           Moonen, De Moor, Vandenberghe, Vandewalle
%           On and off-line identification of linear state space models
%           Intern. Journal of Control, Vol 49, no 1, pp.219-232, 1989
%           
%   Copyright:
%   
%           Peter Van Overschee, December 1995
%           peter.vanoverschee@esat.kuleuven.ac.be
%
%

function [A,B,C,D,ss] = intersec(y,u,i,n,AUXin,sil);

if (nargin < 6);sil = 0;end

mydisp(sil,' ');
mydisp(sil,'   Deterministic Intersection');
mydisp(sil,'   --------------------------');

% Check the arguments
if (nargin < 3);error('intersec needs at least three arguments');end
if (nargin < 4);n = [];end
if (nargin < 5);AUXin = [];end

% Weighting is always empty
W = [];

% Turn the data into row vectors and check
[l,ny] = size(y);if (ny < l);y = y';[l,ny] = size(y);end
[m,nu] = size(u);if (nu < m);u = u';[m,nu] = size(u);end
if (i < 0);error('Number of block rows should be positive');end
if (l < 0);error('Need a non-empty output vector');end
if (m < 0);error('Need a non-empty input vector');end
if (nu ~= ny);error('Number of data points different in input and output');end
if ((nu-2*i+1) < (2*l*i));error('Not enough data points');end

% Determine the number of columns in Hankel matrices
j = nu-2*i+1;

% Check compatibility of AUXin
[AUXin,Wflag] = chkaux(AUXin,i,u(1,1),y(1,1),1,W,sil); 
  
% Compute the R factor
if AUXin == []
  U = blkhank(u/sqrt(j),2*i,j); 		% Input block Hankel
  Y = blkhank(y/sqrt(j),2*i,j); 		% Output block Hankel
  mydisp(sil,'      Computing ... R factor');
  R = triu(qr([U;Y]'))'; 		% R factor
  R = R(1:2*i*(m+l),1:2*i*(m+l)); 	% Truncate
  clear U Y 
else
  R = AUXin(2:2*i*(m+l)+1,:);
  bb = 2*i*(m+l)+1;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%                                  BEGIN ALGORITHM
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Reshuffle R
ax1=kron([1:m:2*i*m]-1,[ones(1,m),zeros(1,l)]) + kron(ones(1,2*i),[1:m,zeros(1,l)]);
ax2=kron([1:l:2*i*l]+2*i*m-1,[zeros(1,m),ones(1,l)]) + kron(ones(1,2*i),[zeros(1,m),1:l]);
ax=ax1+ax2;
R=R(ax,:);

% Compute the SVD
mydisp(sil,'      Computing ... SVD');
[Uh,Sh,Vh]=svd(R);
ss = diag(Sh);

% Determine the order from the singular values
if (n == [])
  ss = ss(2*m*i+1:(2*m+l)*i);
  figure(gcf);hold off;subplot
  [xx,yy] = bar([1:l*i],ss);
  semilogy(xx,yy+10^(floor(log10(min(ss)))));
  axis([0,length(ss)+1,10^(floor(log10(min(ss)))),10^(ceil(log10(max(ss))))]);
  title('Singular Values');
  xlabel('Order');
  n = 0;
  while (n < 1) | (n > l*i-1)
    n = input('      System order ? ');
    if (n == []);n = -1;end
  end
  mydisp(sil,' ');
end


U11=Uh(1:(l+m)*i,1:2*m*i+n);
U12=Uh(1:(l+m)*i,2*m*i+n+1:2*(l+m)*i);
U21=Uh((l+m)*i+1:2*(l+m)*i,1:2*m*i+n);
U22=Uh((l+m)*i+1:2*(l+m)*i,2*m*i+n+1:2*(m+l)*i);
S11=Sh(1:2*m*i+n,1:2*m*i+n);
[uq,sq,vq]=svd(U12'*U11*S11);
uq=uq(:,1:n);
uu=uq'*U12';

% Determine the state matrices
mydisp(sil,['      Computing ... System matrices A,B,C,D (Order ',num2str(n),')']); 
Lhs=[uu*Uh((l+m)+1:(l+m)*(i+1),1:2*m*i+n)*S11;Uh((l+m)*i+m+1:(l+m)*(i+1),1:2*m*i+n)*S11];
Rhs=[uu*Uh(1:(l+m)*i,1:2*m*i+n)*S11;Uh((l+m)*i+1:(l+m)*i+m,1:2*m*i+n)*S11];

% Solve least squares
sol=Lhs/Rhs;

% Extract the system matrices
A = sol(1:n,1:n);
B = sol(1:n,n+1:n+m);
C = sol(n+1:n+l,1:n);
D = sol(n+1:n+l,n+1:n+m);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%                                  END ALGORITHM
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%









