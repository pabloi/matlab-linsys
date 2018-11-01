function [x0,P0,B,D,U,Ud,Ub,opts]=processKalmanOpts(D1,N,aux)
%Argument order: N,x0,P0,B,D,U,opts
%outlierRejection,fastFlag,Ub)

%Defaults:
x0=zeros(D1,1); P0=1e8*eye(D1);  B=0;  D=0;  U=zeros(size(B,2),N); opts=[];
defaultArgs={x0, P0, B, D, U, opts};

%Replace defaults if given and not empty
emptyIdx=cellfun(@isempty,aux);
auxIdx=1:length(aux);
defaultArgs(auxIdx(~emptyIdx))=aux(~emptyIdx); %Replacing defaults with whatever was given
[x0,P0,B,D,opts]=defaultArgs{[1:4,6]};

if ~isfield(opts,'fastFlag') || isempty(opts.fastFlag)
  opts.fastFlag=0;
end
if ~isfield(opts,'outlierFlag') || isempty(opts.outlierFlag)
  opts.outlierFlag=false;
end
if ~isfield(opts,'indD') || isempty(opts.indD)
  opts.indD=1:size(U,1);
end
if ~isfield(opts,'indB') || isempty(opts.indD)
  opts.indB=1:size(U,1);
end
Ud=U(opts.indD,:);
Ub=U(opts.indB,:);
