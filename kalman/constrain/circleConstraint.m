function [H,e,S]=circleConstraint(x)
%x(1)^2+y(1)^2=1
x=x/norm(x);
H=x';
e=1;
S=.01;

%Alt:
%H=diag(sign(x));
%e=sqrt(1-x([2,1]).^2);
%nX=x/norm(x);
%S=.1*(eye(2)-nX*nX'); %No uncertainty along x direction, a lot along orthogonal
%S=S+.01*eye(2); %To give some wiggle room to x
end
