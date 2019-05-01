%
% Demo file for Subspace identification
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

disp('   Welcome to the world of Subspace Identification');
disp('   This demo will illustrate the power of subspace identification')
disp('   algorithms with a simple multivariable example.');
disp(' ')
disp('   Note that in this tutorial demo we consider a simulated example.')
disp('   However, numerous real-life industrial applications can be')
disp('   found under the directory "subspace\applic"');
disp(' ')
disp(' ')
disp('   Hit any key to continue')
pause

clc
echo on
%
%   Consider a multivariable fourth order system a,b,c,d
%   with two inputs and two outputs:

    a = [0.603 0.603 0 0;-0.603 0.603 0 0;0 0 -0.603 -0.603;0 0 0.603 -0.603];
    b = [1.1650,-0.6965;0.6268 1.6961;0.0751,0.0591;0.3516 1.7971];
    c = [0.2641,-1.4462,1.2460,0.5774;0.8717,-0.7012,-0.6390,-0.3600];
    d = [-0.1356,-1.2704;-1.3493,0.9846];
    m = 2; 				% Number of inputs
    l = 2; 				% Number of outputs
      
%   The bode plot:
    w = [0:0.005:0.5]*(2*pi); 		% Frequency vector
    m1 = dbode(a,b,c,d,1,1,w);m2 = dbode(a,b,c,d,1,2,w);
    figure(1);hold off;subplot;clg;
    subplot(221);plot(w/(2*pi),m1(:,1));title('Input 1 -> Output 1');
    subplot(222);plot(w/(2*pi),m2(:,1));title('Input 2 -> Output 1');
    subplot(223);plot(w/(2*pi),m1(:,2));title('Input 1 -> Output 2');
    subplot(224);plot(w/(2*pi),m2(:,2));title('Input 2 -> Output 2');

%   Hit any key
pause
clc

%   We take a white noise sequence of 1000 points as input u.
%   The simulated output is stored in y.

    N = 1000;
    u = randn(N,m);
    y = dlsim(a,b,c,d,u);
    
%   The input and output signals:

    subplot(221);plot(u(:,1));title('Input 1');
    subplot(222);plot(u(:,2));title('Input 2');
    subplot(223);plot(y(:,1));title('Output 1');
    subplot(224);plot(y(:,2));title('Output 2');

%   Hit any key
pause
clc

%   We will now identify this system from the data u,y 
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
	   
%   Hit any key
pause
clc
%
%   The subspace algorithms is now easily started.
%   When prompted for the system order you should enter 4.
%   
    [A,B,C,D] = subid(y,u,i);

%   Did you notice the order was very easy to determine 
%   from the singular values?

%   Hit any key
pause
clc

%   Just to make sure we identified the original system again,
%   we will compare the original and estimated transfer function.
%   
    M1 = dbode(A,B,C,D,1,1,w);
    M2 = dbode(A,B,C,D,1,2,w);
    figure(1)
    hold off;subplot;clg;
    subplot(221);plot(w/(2*pi),[m1(:,1),M1(:,1)]);title('Input 1 -> Output 1');
    subplot(222);plot(w/(2*pi),[m2(:,1),M2(:,1)]);title('Input 2 -> Output 1');
    subplot(223);plot(w/(2*pi),[m1(:,2),M1(:,2)]);title('Input 1 -> Output 2');
    subplot(224);plot(w/(2*pi),[m2(:,2),M2(:,2)]);title('Input 2 -> Output 2');

%   As you can see, the original and identified system are
%   exactly the same.
%   
%   Hit any key
pause
clc

%   This is of-course because the measurements were not noise corrupted.
%   
%   With noise added, the state space system equations become:
%   
%                  x_{k+1) = A x_k + B u_k + K e_k        
%                    y_k   = C x_k + D u_k + e_k
%                 cov(e_k) = R
%                 
    k = [0.1242,-0.0895;-0.0828,-0.0128;0.0390,-0.0968;-0.0225,0.1459]*4;
    r = [0.0176,-0.0267;-0.0267,0.0497];
% 
%   The noise input thus is equal to (the extra chol(r) makes cov(e) = r):

    e = randn(N,l)*chol(r);
% 
%   And the simulated noisy output:

    y = dlsim(a,b,c,d,u) + dlsim(a,k,c,eye(l),e);

%   Hit any key
pause
clc
%
%   Using this output in subid returns a more realistic image of
%   the singular value plot:

    [A,B,C,D] = subid(y,u,i);

%   The order was still visibly equal to 4, wasn't it?

%   Hit any key
pause
clc

%   Again, we compare the identified with the original system.
%   
    M1 = dbode(A,B,C,D,1,1,w);
    M2 = dbode(A,B,C,D,1,2,w);
    figure(1)
    hold off;subplot;clg;
    subplot(221);plot(w/(2*pi),[m1(:,1),M1(:,1)]);title('Input 1 -> Output 1');
    subplot(222);plot(w/(2*pi),[m2(:,1),M2(:,1)]);title('Input 2 -> Output 1');
    subplot(223);plot(w/(2*pi),[m1(:,2),M1(:,2)]);title('Input 1 -> Output 2');
    subplot(224);plot(w/(2*pi),[m2(:,2),M2(:,2)]);title('Input 2 -> Output 2');

%   As you can see, the original and identified system are still very close.
%   
%   Hit any key
pause
clc

%   The function "simul" allows you to check the size of the simulation
%   error.  this is a measure for the difference between the original 
%   and the simulated output:
%   
    [ys,ers] = simul(y,u,A,B,C,D);
%
%   ers contains the error per output in percentage:    
    ers    
%   While ys contains the simulated output:   
    subplot
    plot([y(100:400,1),ys(100:400,1)])
    title('Real (yellow) and simulated (purple) output')

%   They coincide well.    

%   Hit any key
pause
clc

%   The subspace identification function subid also allows you to
%   identify the noise system.  Note that the fourth argument in the 
%   following subid call is the system order.  When this is given, the 
%   singular values are not plotted and you are not prompted for the order.

    [A,B,C,D,K,R] = subid(y,u,i,4);
       
%   Hit any key
pause
clc
%    
%   you can now compute the "one step ahead" prediction using predic:    
    
    [yp,erp] = predic(y,u,A,B,C,D,K);    
    
%   Compare the prediction error and simulation error:
    [ers;erp]
    
%   The prediction errors are significantly smaller.

%   Hit any key
pause
clc

%   In many practical examples, the gap in the singular value plot 
%   is not as clear as in this example.  The order decision then becomes
%   less trivial.  There is however an nice feature of subid which allows
%   for fast computation of systems with different orders. 
%   This can be done through the extra variable AUX which appears
%   as an input as well as an output argument of subid.
%   The last parameter (1) indicates that the algorithm should run silently.
%   
    [A,B,C,D,K,R,AUX] = subid(y,u,i,2,[],[],1);
    era = [];
    for n = 1:6
      [A,B,C,D,K,R] = subid(y,u,i,n,AUX,[],1);
      [ys,ers] = simul(y,u,A,B,C,D);
      era(n,:) = ers;
    end
    
%   Hit any key
pause
clc
%    
%   We have now determined the simulation errors for all systems 
%   from order 1 through 6.
%   Plotting these errors often gives a clearer indication of the order:
%   
    subplot;
    bar([1:6],era);title('Simulation error');
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
    [ersa,erpa] = allord(y,u,i,[1:6],AUX);

%   Hit any key
pause
clc

%   A last feature we would like to illustrate is that subid also
%   can work with principal angles instead of singular values.  
%   The order of the system is then equal to the number of principal
%   angles different from 90 degrees:

    [A,B,C,D,K,R] = subid(y,u,i,[],AUX,'cva');

%   The order is still clearly equal to 4.
%   
%   This concludes the startup demo.  You should now be ready
%   to use subid on your own data.
%   
%   Note that you can also identify time series (no input) with subid.
%   See the help of subid and sto_demo for more explanation.

echo off






