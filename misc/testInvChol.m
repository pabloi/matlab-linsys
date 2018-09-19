S=randn(8,10);
P=S'*S; %psd

%%
tic; for i=1:2000;    iP1=P\eye(size(P)); end;toc
norm(iP1*P-eye(size(P)),'fro')

% This is faster (Slightly) and guarantees PSD
tic;for i=1:2000;    [sP,p]=chol(P);    isP=eye(size(sP))/sP;    iP2=isP*isP';end;toc
if p==0
norm(iP2*P-eye(size(P)),'fro')
end

tic; for i=1:2000;    [~,~,iP3]=pinvchol(P); end;toc

norm(iP3*P-eye(size(P)),'fro')