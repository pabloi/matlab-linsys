# matlab-linsys
Linear dynamic systems toolbox for Matlab.
Includes an implementation of the Kalman filter and Kalman smoother, and several methods to perform system identification. Emphasis on speed.

*Framework:*
The identification methods try to find matrices A,B,C,D,Q,R from a data matrix Y representing N samples (y_k) of a D dimensional output signal, and a matrix U, representing N samples (u_k) of an M dimensional input signal. The system takes the form:
x_{k+1} = Ax_k +Bu_k + w_k
y_k = Cx_k + Du_k + z_k
Where w_k ~ N(0,Q) and z_k ~ N(0,R), and x_k are some (hidden) latent variables.

*Identification methods:* 
sPCA: ONLY identifies a purely deterministic, arbitrary size, LTI-SSM assuming real & different poles, and a constant (single) input. No state noise (i.e. simple least-squares fitting of data).

true EM: an implementation of an Expectation-Maximization algorithm. Alternates between estimating A,B,C,D,Q,R given some guess of the latents x, and estimating x from A,B,C,D,Q,R through the (optimal) Kalman smoother.

fast EM: an approximation of the true EM method, by exploiting steady-state behavior of the kalman filter/smoother

*To Do*
- Implement the extended KF, and extended Kalman Smoother.
- Implement the implicit extended KF.
- Implement the augmented KF.
- Review outlier rejection scheme for KF.
- Implement adaptive KF
- Change approximate logl computation to use median or steady-state uncertainty instead of mean, which makes no sense.
- Change refineQR to use steady-state or median kalman gain instead of mean, which makes no sense.. Check if these changes make a difference.
- Implement a fast loglikelihood calculation that uses the same trick as fast KF: assume that uncertainty reaches a steady-state after some number of steps given by the largest eigenvalue of A.


*Changelist*
v1: added EM based algorithms
