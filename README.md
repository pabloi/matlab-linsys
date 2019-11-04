# matlab-linsys
Linear dynamic systems toolbox for Matlab. Includes tools for identification, selection, visualization, and comparison of linear systems.
Includes an implementation of the Kalman filter and Kalman smoother (in covariance, information and square root formulations), and several methods to perform system identification. Emphasis on speed.

The files within the folder /ext/ are NOT of my authorship, and are included for comparison purposes only. Some of those files have minor modifications for the purposes of their usage in this package. See README files within the specific subfolders for more details.

*Framework:*
The identification methods try to find matrices A,B,C,D,Q,R from a data matrix Y representing N samples (y_k) of a D dimensional output signal, and a matrix U, representing N samples (u_k) of an M dimensional input signal. The system takes the form:
x_{k+1} = Ax_k +Bu_k + w_k
y_k = Cx_k + Du_k + z_k
Where w_k ~ N(0,Q) and z_k ~ N(0,R), and x_k are some (hidden) latent variables.

*Identification methods:*
EM: an implementation of an Expectation-Maximization algorithm. Alternates between estimating A,B,C,D,Q,R given some guess of the latents x, and estimating x from A,B,C,D,Q,R through the (optimal) Kalman smoother. Care was taken to prevent ill-conditioned situations which easily arise on the iteration.
Fast EM: an approximation of the true EM method, by exploiting steady-state behavior of the kalman filter/smoother

SS: implementation of some subspace algorithms as described in van Overschee and de Moor 1996.

*Kalman filtering*
Implementations of stationary Kalman filter and smoother, with emphasis on speed. Speed was achieved by simplifying the equations where possible, using efficient computations for matrix inversions, and exploiting steady-state behavior for stable filters.
Constrained Kalman filter/smoother: this version of the filter adds a step between prediction and update. It enforces a linear constraint of the form H*x=b for the states. This allows to use additional information not captured by system dynamics or measurement equations. By linearizing constraints of the form h(x)=0, it is possible to enforce non-linear constraints too, and even time-varying ones.
The constrained filter allows better handling of unknown dynamics (see testConstrainedKF2.m), in a sense similar to how Lagrangian mechanics deal with reactive forces that impose known links: we forgo the explicit description of the reactive force (i.e. we under/mis-specify the dynamics), and instead recover the true equations of motion/kinematics by enforcing the link we know the force must generate.

*Changelist*
v1: added EM based algorithms
