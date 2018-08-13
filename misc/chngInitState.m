function [B1,D1,X1]=chngInitState(A,B,C,D,X,newX0)
    %Taking an SSM-LTI with some given state trajectories, we can re-define
    %the initial state arbitrarily by modifying B and D, such that we get a
    %modified LTI-SSM with the same output, same A,C matrices but different B,D,x0
    oldX0=X(:,1);
    Dx0=newX0-oldX0;
    B1=B-(A-eye(size(A)))*Dx0;
    D1=D-C*Dx0;
    X1=X+Dx0;
end