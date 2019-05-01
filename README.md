# matlab-linsys
Linear dynamic systems toolbox for Matlab.
Includes an implementation of the Kalman filter and Kalman smoother (in covariance, information and square root formulations), and several methods to perform system identification. Emphasis on speed.

The files within the folder /ext/ are NOT of my authorship, and are included for comparison purposes only. Some of those files have minor modifications for the purposes of their usage in this package. See README files within the specific subfolders for more details.

*Framework:*
The identification methods try to find matrices A,B,C,D,Q,R from a data matrix Y representing N samples (y_k) of a D dimensional output signal, and a matrix U, representing N samples (u_k) of an M dimensional input signal. The system takes the form:
x_{k+1} = Ax_k +Bu_k + w_k
y_k = Cx_k + Du_k + z_k
Where w_k ~ N(0,Q) and z_k ~ N(0,R), and x_k are some (hidden) latent variables.

*Identification methods:*
sPCA: ONLY identifies a purely deterministic, arbitrary size, LTI-SSM assuming real & different poles, and a constant (single) input. No state noise (i.e. simple least-squares fitting of data).

EM: an implementation of an Expectation-Maximization algorithm. Alternates between estimating A,B,C,D,Q,R given some guess of the latents x, and estimating x from A,B,C,D,Q,R through the (optimal) Kalman smoother. Care was taken to prevent ill-conditioned situations which easily arise on the iteration.
Fast EM: an approximation of the true EM method, by exploiting steady-state behavior of the kalman filter/smoother

SS: implementation of some subspace algorithms as described in van Overschee and de Moor 1996.

*Kalman filtering*
Implementations of stationary Kalman filter and smoother, with emphasis on speed. Speed was achieved by simplifying the equations where possible, using efficient computations for matrix inversions, and exploiting steady-state behavior for stable filters.
Constrained Kalman filter/smoother: this version of the filter adds a step between prediction and update. It enforces a linear constraint of the form H*x=b for the states. This allows to use additional information not captured by system dynamics or measurement equations. By linearizing constraints of the form h(x)=0, it is possible to enforce non-linear constraints too, and even time-varying ones.
The constrained filter allows better handling of unknown dynamics (see testConstrainedKF2.m), in a sense similar to how Lagrangian mechanics deal with reactive forces that impose known links: we forgo the explicit description of the reactive force (i.e. we under/mis-specify the dynamics), and instead recover the true equations of motion/kinematics by enforcing the link we know the force must generate.

*To Do*
PRIORITY:
- Figure out why EM algorithm does not seem to converge for problems where dim(output)<dim(state)
- Test alternative canonizations: orthomax, varimax, promax, eyeQ (need to implement the last one)
- Consider alternative initializations of Q,R, or at least mechanisms to escape from local maxima: if either of these matrices are too small with respect to the other, algorithm may get stuck. For example, if Q is small, state uncertainty will be small, and smoothing will have little/no effect when updating states. Consequently, Q will be re-evaluated to a small value in estimateParams, never escaping. Analogous with small R.
Non-priority:
- Allow for partially unknown (diffuse) initial conditions (Follow Durbin and Koopman book, Chapter 5.) -> Is this not equivalent to the information filter currently used?
- Compute a proper (formal) likelihood when initial conditions are diffuse (Durbin and Koopman again).
- Allow for partially unknown/missing data (as opposed to fully missing samples).
- C code: figure out why EM sometimes returns NaN for logL, & why 1 state estimation always fails.
- C code: improve handling of infinite initial uncertainty, right now is just a workaround
- C code: use (modified) chol instead of LU decomp, enforce symmetry of uncertainty matrices.
- logL() computation, and sample rejection for informationFilter2 and informationSmoother.
- Implement fwd/backward algorithm for discrete state markov chains (genKF).
- Create a CircleCI/Docker/Quay integration to continuously test for octave compatibility.
- Implement EM in reduced form when size(C,1)>size(C,2). Kalman smoothing already exploits this, but the M-step could exploit it too: we only need to estimate C'inv(R), rather than a full R.
- Implement GPFA.
- See if we can use EM to reproduce sPCA behavior (this is, identify the best linsys constraining to Q=0)
- Implement the extended KF, and extended Kalman Smoother.
- Implement the implicit extended KF.
- Implement the augmented KF.
- Implement adaptive KF
- Improve constrained KF. Workout how to prevent covariance from collapsing from repeated enforcement of constraints. Figure out how to enforce approximate constraints and non-linear constraints.
- Figure out how to implement constrained smoothing.
- Write a function that automatically generates constraintFunctions as needed by the constrained KF from a nonlinear function handle (that evaluates the function and its gradient).

*Changelist*
v1: added EM based algorithms
