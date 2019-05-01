% 
% [G,L0] = kr2gl(A,K,C,R)
% 
% Description:
%          Find the covariance sequence from
%          the Kalman gain (K) and the innovation covariance (R)
%                
% References:     
%          None
%
% Copyright: 
%          Peter Van Overschee, December 1995
%          peter.vanoverschee@esat.kuleuven.ac.be
%
%

function [G,L0] =  kr2gl(A,K,C,R)

if (K == []) | (R == [])
  G = [];
  L0 = [];
else
  if norm((R+R')/2 - R) > 1e-10;
    error('R should be symmetric');
  end
  % Solve for P
  P = dlyap(A,K*R*K');
  L0 = C*P*C' + R;
  G = K*R + A*P*C';
end

