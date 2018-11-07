function [A,B,C,D,Q,R,X,P]=randomStartEM_par(Y,U,nd,Nreps,opts)

%Handle parallel stuff:
pp=gcp('nocreate');
if isempty(pp) %No parallel pool open, doing regular for, issue warning
    warning('No parallel pool was found, running for loop in serial way.')
    %poolFlag=true;
    parForArg=0; %Argument needed to run parfor serially
    innerLoopSize=1;
    outerLoopSize=Nreps;
else
    parForArg=pp.NumWorkers;
    innerLoopSize=pp.NumWorkers;
    outerLoopSize=ceil(Nreps/innerLoopSize);
end

%First iter:
fprintf(['\n Starting rep 0... \n']);
opts=processEMopts(opts,size(U,1));
outLog=struct();
opt1=opts;
opt1.fastFlag=true; %Enforcing fast filtering
opt1.Niter=500; %Very fast evaluation of initial case, just to get a benchmark.
[A,B,C,D,Q,R,X,P,bestLL,startLog]=EM(Y,U,nd,opt1);
try
    if opts.logFlag
        outLog.opts=opts;
        outLog.startLog=startLog;
        tic;
    end
catch
    warning('EM failed')
    bestLL=-Inf;
end
startLL=bestLL;
[ny,N]=size(Y);
opts.targetLogL=bestLL;

spmd (parForArg)
    for i=1:outerLoopSize %Each worker will go through this loop
    fprintf(['\n Starting rep ' num2str(i) '. Best logL so far=' num2str(bestLL,8) '... \n']);

    %Initialize starting point:
    Xguess=guess(N,nd,ny,opts);

    %Optimize:
    [Ai,Bi,Ci,Di,Qi,Ri,Xi,Pi,logl,repLog]=EM(Y,U,Xguess,opts);
    
    %Get new current best if available: 
    if parForArg>0 && labProbe 
        bestLL=labReceive; 
        opts.targetLogL=bestLL;
    end 
    %If solution improved, save and display:
      if logl>bestLL 
          m.A=Ai; m.B=Bi; m.C=Ci; m.D=Di; m.Q=Qi; m.R=Ri; m.X=Xi; m.P=Pi;
          bestLL=logl;            
          disp(['Success, best logL=' num2str(bestLL,8)])
          if parForArg>0
                labBroadcast(labindex,bestLL);
          end
      end
      if opts.logFlag
          outLog.repLog{i}=repLog;  outLog.repRunTime(i)=nan;
      end
    end
end

%Select best solution:
logl=cell2mat(logl(:));
logl=max(logl,[],2); %Best logl for each parallel worker
[~,idx]=max(logl); %Best across workers
m=m{idx};

if logl(idx)>startLL %Best model is better than original one
    startLL=logl(idx);
    A=m.A;
    B=m.B;
    C=m.C;
    D=m.D;
    Q=m.Q;
    R=m.R;
    X=m.X;
    P=m.P;
end
disp(['End. Best logL=' num2str(startLL)]);

end

function Xguess=guess(N,nd,ny,opts)
    x01=randn(nd,1);
    P01=1e1*eye(nd); %No sense in being certain about made up numbers
    %A1=diag(exp(-1./(.5*N*rand(nd,1)))); %WLOG, diagonal matrix with log-uniformly spaced time-constants
    A1=diag(exp(-1./exp(log(N)*rand(nd,1)))); %WLOG, diagonal matrix with log-uniformly spaced time-constants in the [1,N] interval
    %I think the sign above is unnecessary
    B1=ones(nd,sum(opts.indB~=0)); %WLOG
    Q1=(abs(randn)+1e-4)*eye(nd); %Needs to be psd
    [~,Xsmooth]=fwdSim(U(opts.indB,:),A1,B1,zeros(1,nd),0,[],Q1,[]);
    C1=randn(ny,nd)/ny; %WLOG
    D1=randn(ny,size(U,1));
    %Alt C,D:
    %CD1=Y/[Xsmooth(:,1:end-1);U];
    %C1=CD1(:,1:nd);
    %D1=CD1(:,nd+1:end);
    z=Y-C1*Xsmooth(:,1:end-1)-D1*U(opts.indD,:);
    idx=~any(isnan(z));
    z=z(:,idx);
    R1=z*z'/size(z,2) + C1*Q1*C1'; %Reasonable estimate of R
    %Alt R:
    %R1=(abs(randn)+1e-4)*eye(ny); %Needs to be psd
    [Xguess]=statKalmanSmoother(Y,A1,C1,Q1,R1,x01,P01,B1,D1,U(opts.indD,:),opts);
    Xguess=medfilt1(Xguess,9,[],2); %Some smoothing to avoid starting with very ugly estimates
end
