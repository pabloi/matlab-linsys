% 
%   Deterministic subspace identification (Algorithm 1)
%
%           [A,B,C,D] = det_stat(y,u,i);
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
%           [A,B,C,D,AUX,ss] = det_stat(y,u,i,n,AUX,sil);
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
%              [A,B,C,D] = det_stat(y,u,10,k,AUX);
%           end
%           
%   Reference:
%   
%           Subspace Identification for Linear Systems
%           Theory - Implementation - Applications
%           Peter Van Overschee / Bart De Moor
%           Kluwer Academic Publishers, 1996, Page 52
%           
%   Copyright:
%   
%           Peter Van Overschee, December 1995
%           peter.vanoverschee@esat.kuleuven.ac.be
%
%

function [A,B,C,D,AUX,ss] = det_stat(y,u,i,n,AUXin,sil);

if (nargin < 6);sil = 0;end

mydisp(sil,' ');
mydisp(sil,'   Deterministic algorithm 1');
mydisp(sil,'   -------------------------');

% Check the arguments
if (nargin < 3);error('det_stat needs at least three arguments');end
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
  [U,S,V] = svd(Ob);
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



% **************************************
%               STEP 4 
% **************************************

% Determine gam and gamm
gam  = U1*diag(sqrt(ss(1:n)));
gamm = gam(1:l*(i-1),:);
% And their pseudo inverses
gam_inv  = pinv(gam);
gamm_inv = pinv(gamm);
clear gam gamm



% **************************************
%               STEP 5 
% **************************************

% Compute Obm (the second oblique projection)
Rf = R((2*m+l)*i+l+1:2*(m+l)*i,:); 	% Future outputs
Rp = [R(1:m*(i+1),:);R(2*m*i+1:(2*m+l)*i+l,:)]; % Past (inputs and) outputs
Ru  = R(m*i+m+1:2*m*i,1:mi2); 		% Future inputs
% Perpendicular Future outputs 
Rfp = [Rf(:,1:mi2) - (Rf(:,1:mi2)/Ru)*Ru,Rf(:,mi2+1:2*(m+l)*i)]; 
% Perpendicular Past
Rpp = [Rp(:,1:mi2) - (Rp(:,1:mi2)/Ru)*Ru,Rp(:,mi2+1:2*(m+l)*i)]; 
if (norm(Rpp(:,(2*m+l)*i-2*l:(2*m+l)*i),'fro')) < 1e-10
  Obm  = (Rfp*pinv(Rpp')')*Rp; 		% Oblique projection
else
  Obm = (Rfp/Rpp)*Rp;
end

% Determine the states Xi and Xip
Xi  = gam_inv  * Ob;
Xip = gamm_inv * Obm;
clear gam_inv gamm_inv Obm


% **************************************
%               STEP 6 
% **************************************


% Determine the state matrices
mydisp(sil,['      Computing ... System matrices A,B,C,D (Order ',num2str(n),')']); 
Rhs = [       Xi   ;  R(m*i+1:m*(i+1),:)]; % Right hand side
Lhs = [      Xip   ;  R((2*m+l)*i+1:(2*m+l)*i+l,:)]; % Left hand side

% Solve least squares
sol = Lhs/Rhs;

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





% Make AUX when needed
if nargout > 4
  AUX = zeros((4*l+2*m)*i+1-l,2*(m+l)*i);
  info = [1,i,u(1,1),y(1,1),0]; % in/out - i - u(1,1) - y(1,1) - W
  AUX(1,1:5) = info;
  bb = 1;
  AUX(bb+1:bb+2*(m+l)*i,1:2*(m+l)*i) = R;
  bb = bb+2*(m+l)*i;
  AUX(bb+1:bb+l*i,1:2*(m+l)*i) = Ob;
  bb = bb+l*i;
  AUX(bb+1:bb+l*i,1:l*i) = U;
  AUX(bb+1:bb+l*i,l*i+1) = ss;
end





