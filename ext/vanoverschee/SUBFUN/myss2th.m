% 
% th = myss2th(A,B,C,D,K,flag)
% 
% Description:
%       Converts a state space model to a theta model
%       
%       The state space model is first converted to observability 
%       canonical form.  The parameters of this model are then
%       used in the theta format.
%       
%       If flag = 'oe', and output error form is derived.
%       
% Copyright: 
%          Peter Van Overschee, December 1995
%          peter.vanoverschee@esat.kuleuven.ac.be
%

function th = myss2th(A,B,C,D,K,flag)

% First determine the parameters
% Put everything into observability canonical form
[n,m] = size(B);
[l,n] = size(C);

if (nargin < 6);flag = 'pe';end
if (K == []);flag = 'oe';end

[At,Bt,Ct,Dt,Kt]=ss2obsv(A,B,C,D,K);
% To get a th model, we need to redefine 

if (l < n)
  ap = At(n-l+1:n,:)'; 			% Parameters
  if (K == []) | (flag == 'oe') 
    par=real([ap(:);Bt(:);Dt(:)]);
  else
    par=real([ap(:);Bt(:);Dt(:);Kt(:)]);
  end
else
  ap = At';
  cp = Ct(n+1:l,:)';
  if (K == []) | (flag == 'oe') 
    par=real([ap(:);Bt(:);cp(:);Dt(:)]);
  else
    par=real([ap(:);Bt(:);cp(:);Dt(:);Kt(:)]);
  end
end

aux=[n,m,l];
th=mf2th('x2thf','d',par,aux);



