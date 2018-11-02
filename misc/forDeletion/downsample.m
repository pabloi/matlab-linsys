function [A,B,Q]=downsample(m,A,B,Q)
%Takes a discrete STATIONARY linear SSM and returns a new system that
%predicts takes 1 steps for every m step of the original system.
%See also: upsample

%A=A^m;
oldA=A;
A=expm(m*logm(A)); %This works for non-integer m better (never complex)
B=B/m; 
B=(I-oldA)\((I-A)*B); %Changing B such that the steady state response to a step input is unchanged [presumes a stable system].
Q=Q/sqrt(m);
end