function [Lik,X,V,V1,S] = SmoothLDSMatlab(LDS,Y,U,W)

% function [Lik,X,V,V1,SUMS] = SmoothLDS(LDS,Y)
% function [Lik,X,V,V1,SUMS] = SmoothLDS(LDS,Y,U)
% function [Lik,X,V,V1,SUMS] = SmoothLDS(LDS,Y,U,W)
%
% Kalman Smoothing of a Linear Dynamical System
%
% INPUT:
%
%  LDS is a Linear Dynamical System model with fields:
%
%    A,B,C,D,Q,R,x0,V0
%
%  corresponding to the Model:
%
%    x(t+1) = A*x(t) + B*u(t) + q(t)
%    y(t)   = C*x(t) + D*w(t) + r(t)
%
%    cov([q,r])=[Q 0; 0 R]
%
%    x0    initial state estimate as column vector - (nx)x1
%          (or a cell array with a vector for each "run")
%    V0    cov of the initial state estimate - (nx)x(nx)
%
%  Y     system output
%        for one experiment (or "run"):
%           Y is matrix with outputs as columns - (ny)xT
%        for one or more experiments:
%           Y is cell array of length E containing matrices - (ny)xT(e)
%
%  U     optional matrix with inputs (can be [] to allow for V)
%        for one experiment:
%           U is matrix with outputs as columns - (nu)xT
%        for one or more experiments:
%           U is cell array of length E containing matrices - (nu)xT(e)
%
%  W     optional matrix with "direct through" inputs
%        for one experiment:
%           W is matrix with outputs as columns - (nw)xT
%        for one or more experiments:
%           W is cell array of length E containing matrices - (nw)xT(e)
%
%
% OUTPUT
%
%  Lik   log likelihood: E( log P(Y,X|parameters) | Y,parameters ) (per time step)
%  X     E( x(t) | y(1:T), LDS )
%  V     Cov( x(t) | y(1:T), LDS )
%  V1    Cov( x(t) x(t-1) | y(1:T), LDS )
%
%  For one experiment:
%     X - (nx)xT
%     V - (nx)x(nx)xT
%  For more than one experiment:
%     X,V are cell arrays of length E containing
%       X - (nx)xT(e)
%       V - (nx)x(nx)xT(e)
% SUMS (outer sums)
%   XX= sum x(t) x'(t) + sum V(t,t)
%   YY, XY, UU, XU
%   XX1= sum x(t+1) x'(t) + sum V(t,t+1)
%   XU1= sum x(t+1) u'(t)
%   xxI= sum x(0) x'(0) + sum V(0,0)
%   xxF= sum x(f) x'(f) + sum V(f,f), f= final index
%
% NOTES
%
% This algorithm is the E-step of EM identification algorithm developed
% by Shumway and Stoffer (1982) and Ghahramani and Hinton (1996).


% Copyright (C) 2005 Philip N. Sabes
%
% This program is free software; you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation; either version 2 of the License, or
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program; if not, write to the Free Software
% Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

DEBUGGING = 1;

%%%%%%%%%%%  Dimensions

%% Y
if(~iscell(Y)) 	Y={Y}; cellFlag=0;
else            cellFlag=1;
end

ny = size(Y{1},1);        % Ny=dim of outputs
E  = length(Y);           % E=num of experiments
for e=1:E,
	T(e) = size(Y{e},2);  % T=num of measurements
end

%% x0
if(iscell(LDS.x0))
	nx = length(LDS.x0{1});
	x0 = LDS.x0;          % NX = dim of states
else
	nx = length(LDS.x0);
	for e=1:E, x0{e} = LDS.x0; end
end


%% U
if(nargin>=3)
	if(~iscell(U)) 	U={U}; end
	nu = size(U{1},1);
else
	nu = 0;
end

%% W
if(nargin>=4)
	if(~iscell(W)) 	W={W}; end
	nw = size(W{1},1);
else
	nw = 0;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SMOOTHING (E-STEP)

%%%%% forward pass %%%%%

I  = eye(nx);
A = LDS.A;
C = LDS.C;
Lik = 0;

for e=1:E,

	% Data Structures
	Xp  = zeros(nx,T(e));      % E( x(t)        | y(1),...,y(t-1))
	Vp  = zeros(nx,nx,T(e));   % E( x(t)x(t)'   | y(1),...,y(t-1))
	Xc  = zeros(nx,T(e));      % E( x(t)        | y(1),...,y(t))
	Vc  = zeros(nx,nx,T(e));   % E( x(t)x(t)'   | y(1),...,y(t))
	Xe  = zeros(nx,T(e));      % E( x(t)        | y(1),...,y(T(e)))
	Ve  = zeros(nx,nx,T(e));   % E( x(t)x(t)'   | y(1),...,y(T(e)))
	V1e = zeros(nx,nx,T(e));   % E( x(t)x(t-1)' | y(1),...,y(T(e)))

	Lik = Lik + -0.5*ny*T(e)*log(2*pi);

	for t=1:T(e),

		%% State Update
		if(t>1)
			if(nu>0)  Xp(:,t) = A*Xc(:,t-1) + LDS.B*U{e}(:,t-1);
			else	  Xp(:,t) = A*Xc(:,t-1);
			end
			Vp(:,:,t)= LDS.Q + A*Vc(:,:,t-1)*A';
		else
			Xp(:,1)   = x0{e};
			Vp(:,:,1) = LDS.V0;
        end

        y=Y{e}(:,t);
        if ~any(isnan(y))
		%% Kalman Gain
		invRp = inv( C*Vp(:,:,t)*C' + LDS.R );  % estimate of Cov(Y(t))^-1
		if(DEBUGGING)
			detInvRp = det(invRp);
			if(detInvRp<0) warning('det(invRp) is negative!');
			elseif(detInvRp<1e-8) warning('det(invRp) is poorly conditioned!');
			end
		end
		K = Vp(:,:,t) * C' * invRp;

		%% Innovation
		if(nw>0) innov = y-C*Xp(:,t)-LDS.D*W{e}(:,t);
		else     innov = y-C*Xp(:,t);
		end
		Xc(:,t)   = Xp(:,t) + K*innov;
		Vc(:,:,t) = (I-K*C)*Vp(:,:,t);

		%% LIKELIHOOD
		detInvRp=det(invRp);
		%logDetInvRp=2*sum(log(diag(cRp)));
		if detInvRp < 0
		 	warning('det(invRp) is negative')
		end
		Lik = Lik + 0.5*log(detInvRp) - 0.5*innov'*invRp*innov;
        end

	end % t - forward

	%%%%% backward pass %%%%%

	Xe(:,T(e))    = Xc(:,T(e));
	Ve(:,:,T(e))  = Vc(:,:,T(e));
	if(DEBUGGING)
		dVe = det(Ve(:,:,T(e)));
		if(dVe<0) warning('det(Ve) is negative!');
		elseif(dVe<1e-8) warning('Ve is poorly conditioned!');
		end
	end

	for t=(T(e)-1):-1:1,

		J         = Vc(:,:,t)*A'*inv(Vp(:,:,t+1));
		Xe(:,t)   = Xc(:,t) + J*(Xe(:,t+1)-Xp(:,t+1));
		Ve(:,:,t) = Vc(:,:,t) + J*(Ve(:,:,t+1)-Vp(:,:,t+1))*J';
		if(DEBUGGING)
			dVe = det(Ve(:,:,t));
			if(dVe<0) warning('det(Ve) is negative!');
			elseif(dVe<1e-8) warning('Ve is poorly conditioned!');
			end
		end

		%% Cov(x(t) x(t-1))
		V1e(:,:,t+1) = Ve(:,:,t+1) * J';
		if(DEBUGGING)
			dV1e = det(V1e(:,:,t+1));
			if(abs(dV1e)<1e-8) warning('V1e is poorly conditioned!');
			end
		end

	end % t - backward

	X{e} = Xe;
	V{e} = Ve;
	V1{e} = V1e;

end % e
Lik = Lik/sum(T);


%%%%%%%%%%%%%%% SUMS
clear S
if(nargout>4)
	S.XY=0; S.XX=0; S.YY=0; S.XW=0; S.YW=0; S.WW=0;
	S.UU=0; S.XU=0; S.XX1=0; S.XU1=0; S.xxI=0; S.xxF=0;
	for e=1:E,

		%%%%%%%%%% Sums 1:T
		S.XY = S.XY + X{e}*Y{e}';
		S.XX = S.XX + X{e}*X{e}' + sum(V{e},3);
		S.YY = S.YY + Y{e}*Y{e}';
		if(nw>0)
			S.XW = S.XW + X{e}*W{e}';
			S.YW = S.YW + Y{e}*W{e}';
			S.WW = S.WW + W{e}*W{e}';
		end

		%%%%%%%%%% Sums 2:T
		S.xxI = S.xxI + X{e}(:,1)*X{e}(:,1)' + V{e}(:,:,1);
		S.xxF = S.xxF + X{e}(:,end)*X{e}(:,end)' + V{e}(:,:,end);
		S.XX1 = S.XX1 + X{e}(:,2:end)*X{e}(:,1:(end-1))' + sum(V1{e}(:,:,2:end),3);
		if(nu>0)
			S.UU  = S.UU + U{e}(:,1:(end-1))*U{e}(:,1:(end-1))';
			S.XU  = S.XU + X{e}(:,1:(end-1))*U{e}(:,1:(end-1))';
			S.XU1 = S.XU1 + X{e}(:,2:end)*U{e}(:,1:(end-1))';
		end
	end
end

if(cellFlag==0),
	X=X{1}; V=V{1}; V1= V1{e};
end
