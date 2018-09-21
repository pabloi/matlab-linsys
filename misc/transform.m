function [A,B,C,Q,X,P]=transform(V,A,B,C,Q,X,P)
%All arguments except V are optional. V must be an invertible matrix.
%Equivalent to ss2ss, but allows for more parameters

A=V*A/V;
if nargin>2
    B=V*B;
    if nargin>3
        C=C/V;
        if nargin>4
            Q=V*Q*V';
            if nargin>5
                X=V*X;
                if nargin>6
                    for i=1:size(P,3)
                      P(:,:,i)=V*P(:,:,i)*V'; 
                    end
                end
            end
        end
    end   
end




end