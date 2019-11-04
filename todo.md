# *To Do*:
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
