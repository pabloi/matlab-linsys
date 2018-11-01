function M=processFastFlag(fastFlag,A,N)
  %Determines the number of samples to use for fast filtering
  if isempty(fastFlag) || fastFlag==0 || fastFlag>=N
      M=N; %Do true filtering for all samples
  elseif fastFlag==1
      M2=20; %Default for fast filtering: 20 samples
      M1=ceil(3*max(-1./log(abs(eig(A))))); %This many strides ensures ~convergence of gains before we assume steady-state
      M=max(M1,M2);
      M=min(M,N); %Prevent more than N-1, if this happens, we are not doing fast filtering
  else
      M=min(ceil(abs(fastFlag)),N); %If fastFlag is a number but not 0 or 1, use that as number of samples
      M1=ceil(3*max(-1./log(abs(eig(A))))); %This many strides ensures ~convergence of gains before we assume steady-state
      if M<N && M<M1 %If number of samples provided is not ALL of them, but eigenvalues suggest the system is slower than provided number
          warning('statKSfast:fewSamples','Number of samples for fast filtering were provided, but system time-constants indicate more are needed')
      end
  end
  if M<N && any(abs(eig(A))>1)
      %If the system is unstable, there is no guarantee that the kalman gain
      %converges, and the fast filtering will lead to divergence of estimates
      warning('statKSfast:unstable','Doing steady-state (fast) filtering on an unstable system. States will diverge. Doing traditional filtering instead.')
      M=N;
  end
end
