% 
%   Deterministic subspace identification (Projection)
%
%           [A,B,C,D] = project(y,u,i);
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
%           [A,B,C,D,ss] = project(y,u,i,n,AUX,sil);
%   
%           n:    optional order estimate (default [])
%           AUX:  optional auxilary variable to increase speed (default [])
%           ss:   column vector with singular values
%           sil:  when equal to 1 no text output is generated
%           
%   Example:
%   
%           [A,B,C,D,AUX] = det_alt(y,u,10,2);
%           for k=3:6
%              [A,B,C,D] = project(y,u,10,k,AUX);
%           end
%           
%   Note:
%           The variable AUX is not computed as an output by project.  
%           Variables AUX computed by det_stat or det_alt however can be 
%           used as inputs.
%           
%    Reference:
%   
%           Subspace Identification for Linear Systems
%           Theory - Implementation - Applications
%           Peter Van Overschee / Bart De Moor
%           Kluwer Academic Publishers, 1996, Page 46
%           
%           De moor, Vandewalle
%           A geometrical strategy for the identification of state space models
%           Proc. 3rd Int. Sympos. on Applic. of Multiv. System Techniques, 
%           Plymouth, UK, pp.59-69, April 1987
%
%   Copyright:
%   
%           Peter Van Overschee, December 1995
%           peter.vanoverschee@esat.kuleuven.ac.be
%
%

function [A,B,C,D,ss] = project(y,u,i,n,AUXin,sil);

if (nargin < 6);sil = 0;end

mydisp(sil,' ');
mydisp(sil,'   Deterministic Projection');
mydisp(sil,'   ------------------------');

% Check the arguments
if (nargin < 3);error('project needs at least three arguments');end
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



% First compute the orthogonal projection Yp/Up_perp (past or future does not matter)
Proj = R(2*m*i+1:(2*m+l)*i,m*i+1:(2*m+l)*i);

% Compute the SVD
mydisp(sil,'      Computing ... SVD');
[U,S,V] = svd(Proj);
ss = diag(S);
clear V S Proj


% Determine the order from the singular values
if (n == [])
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

U1 = U(:,1:n); 				% Determine U1


% Sequel is the same as for Deterministic algorithm 2

% **************************************
%               STEP 4 
% **************************************

% Determine gam and gamm
gam  = U1*diag(sqrt(ss(1:n)));
gamm = U1(1:l*(i-1),:)*diag(sqrt(ss(1:n)));
% The pseudo inverse and the orthogonal complement
gam_per  = U(:,n+1:l*i)'; 		% Orthogonal complement
gamm_inv = pinv(gamm); 			% Pseudo inverse




% **************************************
%               STEP 5 
% **************************************

% Determine the matrices A and C
mydisp(sil,['      Computing ... System matrices A,C (Order ',num2str(n),')']); 
A = gamm_inv*gam(l+1:l*i,:);
C = gam(1:l,:);


% **************************************
%               STEP 6 
% **************************************


mydisp(sil,['      Computing ... System matrices B,D (Order ',num2str(n),')']); 
% Determine the matrices M and L
M = gam_per*(R((2*m+l)*i+1:2*(m+l)*i,:)/R(m*i+1:2*m*i,:));
L = gam_per;

% Determine the set of equations
Lhs = zeros(i*(l*i-n),m);
Rhs = zeros(i*(l*i-n),l*i);
aa = 0;
for k=1:i
  Lhs((k-1)*(l*i-n)+1:k*(l*i-n),:) = M(:,(k-1)*m+1:k*m);
  Rhs((k-1)*(l*i-n)+1:k*(l*i-n),1:(i-k+1)*l) = L(:,(k-1)*l+1:l*i);
end
Rhs = Rhs*[eye(l),zeros(l,n);zeros(l*(i-1),l),gamm];


% Solve least squares
sol = Rhs\Lhs;

% Extract the system matrices
B = sol(l+1:l+n,:);
D = sol(1:l,:);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%                                  END ALGORITHM
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

