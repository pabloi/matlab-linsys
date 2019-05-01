%
% Demo file for Stochastic Subspace identification
%
% Copyright: 
%          Peter Van Overschee, December 1995
%          peter.vanoverschee@esat.kuleuven.ac.be
%

clc
disp(' ')
disp(' ')
disp('                SUBSPACE IDENTIFICATION ')
disp('               -------------------------')
disp(' ')

disp('   This file will guide you through the world of time series')
disp('   modeling with subspace identification algorithms.')
disp('  ')
disp('  ')
disp('   Hit any key to continue')
pause

clc
echo on
%
%   Consider a multivariable fourth order system a,k,c,r
%   which is driven by white noise and generates an output y: 
%   
%               x_{k+1} = A x_k + K e_k
%                y_k    = C x_k + e_k
%              cov(e_k) = R

    a = [0.603 0.603 0 0;-0.603 0.603 0 0;0 0 -0.603 -0.603;0 0 0.603 -0.603];
    k = [0.2820,-0.3041;-0.7557,0.0296;0.1919,0.1317;-0.3797,0.6538];
    c = [0.2641,-1.4462,1.2460,0.5774;0.8717,-0.7012,-0.6390,-0.3600];
    r = [0.125274,0.116642;0.116642,0.216978];
    l = 2; 				% Number of outputs
    
%   Hit any key
pause
clc

%   The bode plot of this spectral factor:

    w = [0:0.005:0.5]*(2*pi); 		% Frequency vector
    m1 = dbode(a,k,c,eye(l),1,1,w);
    m2 = dbode(a,k,c,eye(l),1,2,w);
    figure(1)
    hold off;subplot;clg;
    subplot(221);plot(w/(2*pi),m1(:,1));title('Output 1 -> Output 1');
    subplot(222);plot(w/(2*pi),m2(:,1));title('Output 2 -> Output 1');
    subplot(223);plot(w/(2*pi),m1(:,2));title('Output 1 -> Output 2');
    subplot(224);plot(w/(2*pi),m2(:,2));title('Output 2 -> Output 2');

%   Hit any key
pause
clc
%
%   We take a white noise sequence of 4000 points as (unmeasured) input e.
%   The simulated output is stored in y. (chol(r) makes cov(e) = r).

    N = 4000;
    e = randn(N,l)*chol(r);
    y = dlsim(a,k,c,eye(l),e);
    
%   The output signals:

    subplot(211);plot(y(:,1));title('Output 1');
    subplot(212);plot(y(:,2));title('Output 2');

%   Hit any key
pause
clc

%   We will now identify this system from the data y 
%   with the subspace identification algorithm: subid
%   
%   The only extra information we need is the "number of block rows" i
%   in the block Hankel matrices.  This number is easily determined
%   as follows:

%   Say we don't know the order, but think it is maximally equal to 10.
%   
%       max_order = 10;
%   
%   As described in the help of subid we can determine "i" as follows:
%   
%       i = 2*(max_order)/(number of outputs)
%       
        i = 2*(10)/2;
%
%   Hit any key
pause
clc
%
%   The subspace algorithms is now easily started.
%   Note the dummy outputs (du1 and du2) where normally the B and D 
%   matrices are located.  Also note that u = [];
%   When prompted for the system order you should enter 4.
%   
    [A,du1,C,du2,K,R] = subid(y,[],i);

%   Did you notice the order was very easy to determine 
%   from the number of principal angles different from 90 degrees?

%   Hit any key
pause
clc

%   Just to make sure we identified the original system again,
%   we will compare the original and estimated transfer function.
%   
    M1 = dbode(A,K,C,eye(l),1,1,w);
    M2 = dbode(A,K,C,eye(l),1,2,w);
    figure(1)
    hold off;subplot;clg;
    subplot(221);plot(w/(2*pi),[m1(:,1),M1(:,1)]);title('Output 1 -> Output 1');
    subplot(222);plot(w/(2*pi),[m2(:,1),M2(:,1)]);title('Output 2 -> Output 1');
    subplot(223);plot(w/(2*pi),[m1(:,2),M1(:,2)]);title('Output 1 -> Output 2');
    subplot(224);plot(w/(2*pi),[m2(:,2),M2(:,2)]);title('Output 2 -> Output 2');

%   As you can see, the original and identified spectral factor are close
%   
%   Hit any key
pause
clc

%   The function "predic" allows you to check the size of the prediction
%   error.  This is a measure for the difference between the original 
%   and the predicted output:
%   
    [yp,erp] = predic(y,[],A,[],C,[],K);
%
%   erp contains the error per output in percentage:    
    erp    
%   While yp contains the predicted output: 
    subplot
    plot([y(100:400,1),yp(100:400,1)])
    title('Real (yellow) and predicted (purple) output')

%   They coincide well.    
%   Hit any key
pause
clc

%   In many practical examples, the gap in the singular value plot 
%   is not as clear as in this example.  The order decision then becomes
%   less trivial.  There is however an nice feature of sto_pos which allows
%   for fast computation of systems with different orders. 
%   This can be done through the extra variable AUX which appears
%   as an input as well as an output argument of subid.
%   The last parameter (1) indicates that the algorithm should run silently.
  
    [A,du1,C,du2,K,R,AUX] = subid(y,[],i,2,[],[],1);
    era = [];
    for n = 1:6
      [A,B,C,D,K,R] = subid(y,[],i,n,AUX,[],1);
      [yp,erp] = predic(y,[],A,[],C,[],K);
      era(n,:) = erp;
    end
    
%   Hit any key
pause
clc
%           
%   We have now determined the prediction errors for all systems 
%   from order 1 through 6.
%   Plotting these often gives a clearer indication of the order:
%   
    subplot;
    bar([1:6],era);title('Prediction error');
    xlabel('System order');
    
%   It now even becomes more clear that the system order is 4.
%    
%   Hit any key
pause
clc
%   
%   We did find this very useful, so we included the above code in 
%   a function: allord.  The above result could have been obtained 
%   with the one line code:
%   
    [ersa,erpa] = allord(y,[],i,[1:6],AUX);

%   Hit any key
pause
clc

%   A last feature we would like to illustrate is that subid (or
%   the other stochastic identification algorithms) also
%   can work with another basis.     
%   The order of the system is then equal to the number of singular
%   values different from zero (instead of the number of angles
%   different from 90 degrees):

    [A,du1,C,du2,K,R] = subid(y,[],i,[],AUX,'sv');

%   The order is still clearly equal to 4.
%   
%   This concludes the stochastic demo.  You should now be ready
%   to use subid on your own data.
%   
%   Note that other time series identification algorithms are:
%   sto_stat, sto_alt and sto_pos.
%   
%   For a demonstration of combined identification, see sta_demo.
echo off






