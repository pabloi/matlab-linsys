% 
%   Combined subspace identification (Algorithm 1)
%
%           [A,B,C,D,K,R] = com_alt(y,u,i);
% 
%   Inputs:
%           y: matrix of measured outputs
%           u: matrix of measured inputs
%           i: number of block rows in Hankel matrices
%              (i * #outputs) is the max. order that can be estimated 
%              Typically: i = 2 * (max order)/(#outputs)
%           
%   Outputs:
%           A,B,C,D,K,R: combined state space system
%           
%                  x_{k+1) = A x_k + B u_k + K e_k        
%                    y_k   = C x_k + D u_k + e_k
%                 cov(e_k) = R
%   Optional:
%
%           [A,B,C,D,K,R,AUX,ss] = com_alt(y,u,i,n,AUX,W,sil);
%   
%           n:    optional order estimate (default [])
%           AUX:  optional auxilary variable to increase speed (default [])
%           W:    optional weighting flag
%                    N4SID: Numerical algo. for State Space  
%                           Subspace System ID (default)
%                    MOESP: Multivar. Output-Error State Space
%                    CVA:   Canonical variable analysis
%           ss:   column vector with singular values
%           sil:  when equal to 1 no text output is generated
%           
%   Example:
%   
%           [A,B,C,D,K,R,AUX] = com_alt(y,u,10,2);
%           for k=3:6
%              [A,B,C,D] = com_alt(y,u,10,k,AUX);
%           end
%           
%   Reference:
%   
%           Subspace Identification for Linear Systems
%           Theory - Implementation - Applications
%           Peter Van Overschee / Bart De Moor
%           Kluwer Academic Publishers, 1996, Page 121 (Fig 4.6)
%           
%   Copyright:
%   
%           Peter Van Overschee, December 1995
%           peter.vanoverschee@esat.kuleuven.ac.be
%
%

function [A,B,C,D,K,Ro,AUX,ss] = com_alt(y,u,i,n,AUXin,W,sil);

if (nargin < 7);sil = 0;end

mydisp(sil,' ');
mydisp(sil,'   Combined algorithm 1');
mydisp(sil,'   --------------------');

% Check the arguments
if (nargin < 3);error('com_alt needs at least three arguments');end
if (nargin < 4);n = [];end
if (nargin < 5);AUXin = [];end
if (nargin < 6);W = [];end
if (W == []);W = 'N4SID';end

% Turn the data into row vectors and check
[l,ny] = size(y);if (ny < l);y = y';[l,ny] = size(y);end
[m,nu] = size(u);if (nu < m);u = u';[m,nu] = size(u);end
if (i < 0);error('Number of block rows should be positive');end
if (l < 0);error('Need a non-empty output vector');end
if (m < 0);error('Need a non-empty input vector');end
if (nu ~= ny);error('Number of data points different in input and output');end
if ((nu-2*i+1) < (2*l*i));error('Not enough data points');end
Wn = 0;
if (length(W) == 5) 
  if (prod(W == 'N4SID') | prod(W == 'n4sid') | prod(W == 'N4sid'));Wn = 1;end 
  if (prod(W == 'MOESP') | prod(W == 'moesp') | prod(W == 'Moesp'));Wn = 2;end
end    
if (length(W) == 3) 
  if (prod(W == 'CVA') | prod(W == 'cva') | prod(W == 'Cva'));Wn = 3;end 
end
if (Wn == 0);error('W should be N4SID, MOESP or CVA');end
W = Wn;

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




% **************************************
%               STEP 1 
% **************************************

mi2 = 2*m*i;
% Set up some matrices
if (AUXin == []) | (Wflag == 1)
  Rf = R((2*m+l)*i+1:2*(m+l)*i,:); 	% Future outputs
  Rp = [R(1:m*i,:);R(2*m*i+1:(2*m+l)*i,:)]; % Past (inputs and) outputs
  Ru  = R(m*i+1:2*m*i,1:mi2); 		% Future inputs
  % Perpendicular Future outputs 
  Rfp = [Rf(:,1:mi2) - (Rf(:,1:mi2)/Ru)*Ru,Rf(:,mi2+1:2*(m+l)*i)]; 
  % Perpendicular Past
  Rpp = [Rp(:,1:mi2) - (Rp(:,1:mi2)/Ru)*Ru,Rp(:,mi2+1:2*(m+l)*i)]; 
end

% The oblique projection:
% Computed as on page 166 Formula 6.1
% obl/Ufp = Yf/Ufp * pinv(Wp/Ufp) * (Wp/Ufp)

if (AUXin == [])
  % Funny rank check (SVD takes too long)
  % This check is needed to avoid rank deficiency warnings
  if (norm(Rpp(:,(2*m+l)*i-2*l:(2*m+l)*i),'fro')) < 1e-10
    Ob  = (Rfp*pinv(Rpp')')*Rp; 	% Oblique projection
  else
    Ob = (Rfp/Rpp)*Rp;
  end
else
  % Determine Ob from AUXin
  Ob = AUXin(bb+1:bb+l*i,1:2*(l+m)*i);
  bb = bb+l*i;
end


% **************************************
%               STEP 2 
% **************************************

% Compute the SVD
if (AUXin == []) | (Wflag == 1)
  mydisp(sil,'      Computing ... SVD');
  % Compute the matrix WOW we want to take an SVD of
  % W = 1 (N4SID), W = 2 (MOESP), W = 3 (CVA)
  if (W == 1)
    WOW = Ob;
  else
    % Moesp or CVA: extra projection
    % Extra projection of Ob on Uf perpendicular
    WOW = [Ob(:,1:mi2) - (Ob(:,1:mi2)/Ru)*Ru,Ob(:,mi2+1:2*(m+l)*i)];
    if (W == 3)
      % Extra weighting for CVA
      W1i = triu(qr(Rf'));
      W1i = W1i(1:l*i,1:l*i)';
      WOW = W1i\WOW;
    end
  end
  [U,S,V] = svd(WOW);
  if W == 3;U = W1i*U;end 		% CVA
  ss = diag(S);
  clear V S WOW
else
  U = AUXin(bb+1:bb+l*i,1:l*i);
  ss = AUXin(bb+1:bb+l*i,l*i+1);
end



% **************************************
%               STEP 3 
% **************************************

% Determine the order from the singular values
if (n == [])
  figure(gcf);hold off;subplot
  if (W == 3)
    bar([1:l*i],real(acos(ss))*180/pi);
    title('Principal Angles');
    ylabel('degrees');
  else
    [xx,yy] = bar([1:l*i],ss);
    semilogy(xx,yy+10^(floor(log10(min(ss)))));
    axis([0,length(ss)+1,10^(floor(log10(min(ss)))),10^(ceil(log10(max(ss))))]);
    title('Singular Values');
  end
  xlabel('Order');
  n = 0;
  while (n < 1) | (n > l*i-1)
    n = input('      System order ? ');
    if (n == []);n = -1;end
  end
  mydisp(sil,' ');
end

U1 = U(:,1:n); 				% Determine U1


% **************************************
%               STEP 4 
% **************************************

% Determine gam and gamm
gam  = U1*diag(sqrt(ss(1:n)));
gamm = gam(1:l*(i-1),:);
% The pseudo inverses
gam_inv  = pinv(gam); 			% Pseudo inverse
gamm_inv = pinv(gamm); 			% Pseudo inverse


% **************************************
%               STEP 5 
% **************************************

% Determine the matrices A and C
mydisp(sil,['      Computing ... System matrices A,C (Order ',num2str(n),')']); 
Rhs = [   [gam_inv*R((2*m+l)*i+1:2*(m+l)*i,1:(2*m+l)*i),zeros(n,l)] ; ...
    R(m*i+1:2*m*i,1:(2*m+l)*i+l)];
Lhs = [        gamm_inv*R((2*m+l)*i+l+1:2*(m+l)*i,1:(2*m+l)*i+l) ; ...
    R((2*m+l)*i+1:(2*m+l)*i+l,1:(2*m+l)*i+l)];

% Solve least square
sol = Lhs/Rhs;

% Extract the system matrices A and C
A = sol(1:n,1:n);
C = sol(n+1:n+l,1:n);


% **************************************
%               STEP 6 
% **************************************


mydisp(sil,['      Computing ... System matrices B,D (Order ',num2str(n),')']); 
% Use formula 4.53 on page 119
L1 = A * gam_inv;
L2 = C * gam_inv;
M  = [zeros(n,l),gamm_inv];
X  = [eye(l),zeros(l,n);zeros(l*(i-1),l),gamm];

for k=1:i
  % Calculate N1, N2
  N1((k-1)*n+1:k*n,:)=...
      [M(:,(k-1)*l+1:l*i)-L1(:,(k-1)*l+1:l*i),zeros(n,(k-1)*l)];
  N2((k-1)*l+1:k*l,:)=...
      [-L2(:,(k-1)*l+1:l*i),zeros(l,(k-1)*l)];

  if k == 1;N2(1:l,1:l) = eye(l) + N2(1:l,1:l);end

  % kap1 and kap2
  kap1=[kap1; sol(1:n,n+(k-1)*m+1:n+k*m)];
  kap2=[kap2; sol(n+1:n+l,n+(k-1)*m+1:n+k*m)];
end

% Solve least squares
sol_bd = ([N1;N2]*X)\[kap1;kap2];

% Get the system matrices out
D = sol_bd(1:l,:);
B = sol_bd(l+1:l+n,:);


% **************************************
%               STEP 7 
% **************************************

% Determine QSR from the residuals
mydisp(sil,['      Computing ... System matrices G,L0 (Order ',num2str(n),')']); 
% Determine the residuals
res = Lhs - sol*Rhs; 			% Residuals
cov = res*res'; 			% Covariance
Qs = cov(1:n,1:n);Ss = cov(1:n,n+1:n+l);Rs = cov(n+1:n+l,n+1:n+l); 

sig = dlyap(A,Qs);
G = A*sig*C' + Ss;
L0 = C*sig*C' + Rs;

% Determine K and Ro
mydisp(sil,'      Computing ... Riccati solution')
[K,Ro] = gl2kr(A,G,C,L0);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%                                  END ALGORITHM
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





% Make AUX when needed
if nargout > 6
  AUX = zeros((4*l+2*m)*i+1,2*(m+l)*i);
  info = [1,i,u(1,1),y(1,1),W]; % in/out - i - u(1,1) - y(1,1) - W
  AUX(1,1:5) = info;
  bb = 1;
  AUX(bb+1:bb+2*(m+l)*i,1:2*(m+l)*i) = R;
  bb = bb+2*(m+l)*i;
  AUX(bb+1:bb+l*i,1:2*(m+l)*i) = Ob;
  bb = bb+l*i;
  AUX(bb+1:bb+l*i,1:l*i) = U;
  AUX(bb+1:bb+l*i,l*i+1) = ss;
end





