%
% Application of Subspace Identification to:
%
%           A laboratory dryer
%
%       
% Reference:
%   
%           Subspace Identification for Linear Systems
%           Theory - Implementation - Applications
%           Peter Van Overschee / Bart De Moor
%           Kluwer Academic Publishers, 1996, Page 189
%           
% Data:
%           The data is contained in the file appl2.mat
%           We are grateful to Lennart Ljung for providing us the data.
%          
% Copyright: 
%          Peter Van Overschee, December 1995
%          peter.vanoverschee@esat.kuleuven.ac.be
%

clc;
disp(' ');
disp(' ');
disp('               Subspace Identification ')
disp('                        of ')
disp('                 A laboratory dryer');
disp('                --------------------');
disp(' ');

% Load the data
load appl2.mat

ti = 0.5;
% Preprocess a bit
u=dtrend(u);y=dtrend(y);

% Some parameters:
m = min(size(u)); 			% Number of inputs
l = min(size(y)); 			% Number of outputs
ax_id = [1:500];n_id = length(ax_id); 	% Identification axis
ax_val = [501:1000];n_val = length(ax_val); % Validation axis
u_id = u(ax_id,:);y_id = y(ax_id,:); 	% Identification data
u_val = u(ax_val,:);y_val = y(ax_val,:); % Validation data
i = 15; 				% Number of block rows

% Display the parameters
disp(['     Number of inputs:             ',num2str(m)]);
disp(['     Number of outputs:            ',num2str(l)]);
disp(['     Number of ID data points:     ',num2str(n_id)]);
disp(['     Number of VAL data points:    ',num2str(n_val)]);
disp(['     Number of block rows:         ',num2str(i)]);
disp(['     Total Computation (Pentium):  ',num2str(ti),' min']);
disp(' ')
disp(' ')
disp('     Suggested order is:           4') 
disp(' ')
disp('     Hit any key to continue');
pause

tic
% Subspace identification
[A,B,C,D,K,R,AUX] = subid(y_id,u_id,i,[]);

% Other subspace
[n,n] = size(A);
[A1,B1,C1,D1,K1,R1] = com_alt(y_id,u_id,i,n,AUX,[],1);
[A2,B2,C2,D2,K2,R2] = com_stat(y_id,u_id,i,n,AUX,[],1);
tt1 = toc;

% Compute the simulation errors
[du,ers1_id] = simul(y,u,A1,B1,C1,D1,ax_id);
[du,ers2_id] = simul(y,u,A2,B2,C2,D2,ax_id);
[du,ers3_id] = simul(y,u,A,B,C,D,ax_id);
[du,ers1_val] = simul(y,u,A1,B1,C1,D1,ax_val);
[du,ers2_val] = simul(y,u,A2,B2,C2,D2,ax_val);
[du,ers3_val] = simul(y,u,A,B,C,D,ax_val);

% And the prediction errors
[du,erk1_id] = predic(y,u,A1,B1,C1,D1,K1,ax_id);
[du,erk2_id] = predic(y,u,A2,B2,C2,D2,K2,ax_id);
[du,erk3_id] = predic(y,u,A,B,C,D,K,ax_id);
[du,erk1_val] = predic(y,u,A1,B1,C1,D1,K1,ax_val);
[du,erk2_val] = predic(y,u,A2,B2,C2,D2,K2,ax_val);
[du,erk3_val] = predic(y,u,A,B,C,D,K,ax_val);


% Use A,B,C,D,K as initial guesses for a pem and an OE routine:
disp('  ')
disp('  ')
disp('  ')
disp('      This concludes the subspace identification part');
disp('  ')
disp('      We will now use the subspace model as an initial')
disp('      guess for the optimization routines of the system')
disp('      identification toolbox: pem and oe');
disp('      This may take up to 0.5 minute')  
disp(' ')  
disp('      Hit any key to continue');
disp(' ')
pause

% Prediction error method Ljung
thi_pem = myss2th(A,B,C,D,K);
thi_oe = myss2th(A,B,C,D,[],'oe');

tic
th_pem = pem([y_id,u_id],thi_pem);
th_oe = pem([y_id,u_id],thi_oe);
tt2 = toc;

% Convert to state space
[Ap,Bp,Cp,Dp,Kp] = th2ss(th_pem);
[Ao,Bo,Co,Do,Ko] = th2ss(th_oe);

% Compute the simulation errors
[du,ersp_id] = simul(y,u,Ap,Bp,Cp,Dp,ax_id);
[du,erso_id] = simul(y,u,Ao,Bo,Co,Do,ax_id);
[du,ersp_val] = simul(y,u,Ap,Bp,Cp,Dp,ax_val);
[du,erso_val] = simul(y,u,Ao,Bo,Co,Do,ax_val);

% And the prediction errors
[du,erkp_id] = predic(y,u,Ap,Bp,Cp,Dp,Kp,ax_id);
[du,erko_id] = predic(y,u,Ao,Bo,Co,Do,Ko,ax_id);
[du,erkp_val] = predic(y,u,Ap,Bp,Cp,Dp,Kp,ax_val);
[du,erko_val] = predic(y,u,Ao,Bo,Co,Do,Ko,ax_val);


% And show the results
show_res('Labo Dryer',m,l,n_id,n_val,...
    erk3_id,erk3_val,ers3_id,ers3_val,...
    erk1_id,erk1_val,ers1_id,ers1_val,...
    erk2_id,erk2_val,ers2_id,ers2_val,...
    erkp_id,erkp_val,ersp_id,ersp_val,...
    erko_id,erko_val,erso_id,erso_val,tt1,tt2,n);
    




