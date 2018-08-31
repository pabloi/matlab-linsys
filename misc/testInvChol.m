S=randn(10);
P=S'*S; %psd

%%
tic
for i=1:2000
    iP=P\eye(size(P));
end
toc

%% This is faster (Slightly) and guarantees PSD
tic
for i=1:2000
    sP=chol(P);
    isP=sP\eye(size(sP));
    iP=isP*isP';
end
toc