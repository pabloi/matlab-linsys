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
                if isa(X,'cell')
                    for i=1:length(X)
                        X{i}=V*X{i};
                    end
                else
                    X=V*X;
                end
                if nargin>6
                    if isa(P,'cell')
                        for k=1:length(P)
                           for i=1:size(P{k},3)
                                P{k}(:,:,i)=V*P{k}(:,:,i)*V'; 
                           end
                        end
                    else
                        for i=1:size(P,3)
                          P(:,:,i)=V*P(:,:,i)*V'; 
                        end
                    end
                end
            end
        end
    end   
end




end