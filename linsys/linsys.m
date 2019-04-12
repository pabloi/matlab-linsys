classdef linsys
    %linsys This defines a class to describe linear models through their
    %state-space representation. Mostly consists of a bunch of wrappers to
    %more easily use the functions included in this toolbox (EM
    %identification, Kalman filtering and smoothing, visualization, etc.)

    properties
        A
        B
        C
        D
        Q
        R
        name='';
    end
    properties (Dependent)
        order %Model order, number of stats
        Ninput %Number of inputs
        Noutput %Number of outputs
        hash
    end

    methods
        function this = linsys(A,C,R,B,D,Q) %Constructor
            if size(A,1)~=size(A,2)
                error('A matrix is not square')
            end
            if size(Q,1)~=size(Q,2)
                error('Q matrix is not square')
            end
            if size(R,1)~=size(R,2)
                error('R matrix is not square')
            end
            if size(B,1) ~=size(A,1)
                error('A and B have inconsistent sizes')
            end
            if size(C,1) ~= size(D,1)
                error('Inconsistent C and D')
            end
            if size(C,1) ~= size(R,1)
                error('Inconsistent C and R')
            end
            if size(Q,1) ~=size(A,1)
                error('A and Q have inconsistent sizes')
            end
            if size(B,2) ~= size(D,2)
                error('Inconsistent input sizes')
            end
            this.A=A;
            this.C=C;
            this.R=R;
            this.B=B;
            this.D=D;
            this.Q=Q;
        end
        function dfit=fit(this,datSet,initC,method)
            if nargin<3
                initC=[];
            end
            if nargin<4 || isempty(method)
                method=[];
            end
            dfit=dataFit(this,datSet,method,initC);
        end
        function [filteredState,oneAheadState,rejSamples,logL] = Kfilter(this,datSet,initC,opts)
            if nargin<4
                opts=[];
            end
            if isa(datSet,'cell') %Multiple realizations
                if nargin<3
                    initC=cell(size(datSet));
                end
                for i=1:length(datSet)
                    [X{i},P{i},Xp{i},Pp{i},rejSamples{i},logL(i)] = Kfilter(this,datSet{i},initC{i},opts);
                end
            else
                if nargin<3 || isempty(initC)
                    initC=initCond([],[]);
                end
                [X,P,Xp,Pp,rejSamples,logL]=statKalmanFilter(datSet.out,this.A,this.C,this.Q,this.R,initC.state,initC.covar,this.B,this.D,datSet.in,opts);
                filteredState=stateEstimate(X,P);
                oneAheadState=stateEstimate(Xp,Pp);
            end
        end
        function [smoothState,filteredState,oneAheadState,rejSamples,logL] = Ksmooth(this,datSet,initC,opts)
            if nargin<4
                opts=[];
            end
            if isa(datSet,'cell') %Multiple realizations
                if nargin<3
                    initC=cell(size(datSet));
                end
                for i=1:length(datSet)
                    [X{i},P{i},Pt{i},Xf{i},Pf{i},Xp{i},Pp{i},rejSamples{i},logL(i)] = Ksmooth(this,datSet{i},initC{i},opts);
                end
            else
                unusedSamp=0;
                if nargin<3 || isempty(initC)
                    initC=initCond(zeros(this.order,0)); %Improper prior: can be problematic
                end
                [X,P,Pt,Xf,Pf,Xp,Pp,rejSamples,logL]=statKalmanSmoother(datSet.out,this.A,this.C,this.Q,this.R,initC.state,initC.covar,this.B,this.D,datSet.in,opts);
                smoothState=stateEstimate(X,P,Pt);
                filteredState=stateEstimate(Xf,Pf);
                oneAheadState=stateEstimate(Xp,Pp);
            end
        end
        function stateE=predict(this,stateE,in)
            %Predicts new states for N samples in the future, given current state
            if nargin<3 %Projecting a single stride in the future, with null input
                in=zeros(this.Ninput,1);
            end
            x=zeros(size(stateE.state));
            P=zeros(size(stateE.uncertainty));
            for j=1:stateE.Nsamp
                x(:,j)=stateE.state(:,j);
                P(:,:,j)=stateE.uncertainty(:,:,j);
                for i=1:size(in,2);
                    x(:,j)=this.A*x(:,j) + this.B*in(:,i); %Project a single step into the future
                    P(:,:,j)=this.A*P(:,:,j)*this.A' + this.Q;
                end
            end
        end
        function [datSet,stateE]=simulate(this,input,initC,deterministicFlag,noiselessFlag)
            %Simulates a realization of the modeled system given the initial
            %conditions (uncertainty in inital condition is ignored)
            %If deterministicFlag=true, then Q is set to 0 before simulating
            %(there is still observational noise, but states evolve in a deterministic way)
            if nargin>3 && deterministicFlag
                this.Q=zeros(size(this.Q));
            end
            if nargin>4 && noiselessFlag
                this.R=zeros(size(this.R));
            end
            if nargin<3 || isempty(initC)
                iC=[];
            else
                iC=initC.state;
            end
            [out,state]=fwdSim(input,this.A,this.B,this.C,this.D,iC,this.Q,this.R);
            datSet=dset(input,out);
            stateE=stateEstimate(state,zeros(this.order));
        end
        function mdl=linsys2struct(this)
            warning('off'); %Prevents complaint about making structs from objects
            mdl=struct(this);
            mdl.J=mdl.A;
            warning('on');
        end
        function fh=viz(this)
            [fh] = vizModels({this.linsys2struct});
        end
        function [this,V]=canonize(this,method)
          if nargin<2
            method=[];
          end
          [this.A,this.B,this.C,~,V,this.Q,~] = canonize(this.A,this.B,this.C,[],this.Q,[],method);
        end
        function [this]=transform(this,V)
            [this.A,this.B,this.C,this.Q]=transform(V,this.A,this.B,this.C,this.Q);
        end
        function newThis=upsample(this,Nfactor)
            error('unimplemented')
           newThis=this; %Doxy
        end
        function newThis=downsample(this,Nfactor)
            error('unimplemented')
           newThis=this; %Doxy
        end
        function [fh,fh2]=vizFit(this,datSet,initC)
          [fh,fh2]=datSet.vizFit(this);
        end
        function fh=vizRes(this,datSet,initC)
            [fh]=datSet.vizRes(this);
        end
        function l=logL(this,datSet,initC)
            %Per sample, per dim of output
            if nargin<3
                initC=initCond([],[]);
            end
            l=dataLogLikelihood(datSet.out,datSet.in,this.A,this.B,this.C,this.D,this.Q,this.R,initC.state,initC.covar,'exact');
            l=l;
        end
        function ord=get.order(this)
            ord=size(this.A,1);
        end
        function ord=get.Ninput(this)
            ord=size(this.B,2);
        end
        function ord=get.Noutput(this)
            ord=size(this.C,1);
        end
        function M=noiseCovar(this,N)
           %Computes the covariance of the stochastic component of the states after N steps
           I=eye(size(this.Q));
           M=this.Q;
           A=this.A;
           for i=1:N-1; M=A*M*A'+this.Q; end
        end
        function X=detPredict(this,N)
            %Computes the deterministic component of the state after N steps starting from null initial
            %condition, assuming a step input in the first input.
            A=this.A;
            I=eye(size(this.Q));
            B=this.B;
            X=(I-A)\(I-A^N)*B(:,1);
        end
        function s=SNR(this,N)
           %Estimates a SNR-like covariance estimate
           X=this.detPredict(N);
           M=this.noiseCovar(N);
           %s=X'*M*X;
           s=(X.^2)./diag(M);
        end
        function newThis=pad(this,padIdx,Dpad,Cpad,Rpad)
            %This takes a model and expands its output by padding C,D,R
            %If not given, C and D are padded with 0, R is padded with the infinite variances
            %If given, C has to be length(padIdx) x this.order
            %D has to be length(padIdx) x this.Ninput
            %R has to be length(padIdx) x 1 (vector representing diagonal entries only)
            %Check: padIdx is an integer vector (not boolean), without repeats
            %To do: add a method to populate C,D,R with appropriate (MLE?)
            %values. This requires a datSet for state estimation and fitting.
            if islogical(padIdx) %Transform boolean to index list
                padIdx=find(padIdx);
            end
            if any(padIdx==0) || length(unique(padIdx))~=length(padIdx)
              error('')
            end
            %To Do: check that C,D do not contain NaN, R does not contain Nan or <=0.
            newThis=this;
            newIdx=length(padIdx);
            oldIdx=this.Noutput;
            newSize=newIdx+oldIdx;
            newThis.C=nan(newSize,this.order);
            newThis.D=nan(newSize,this.Ninput);
            newThis.R=zeros(newSize);
            if nargin<4 ||isempty(Cpad)
              Cpad=zeros(newIdx,this.order);
            end
            if nargin<3 || isempty(Dpad)
              Dpad=zeros(newIdx,this.Ninput);
            end
            if nargin<5 || isempty(Rpad)
              Rpad=Inf(newIdx,1);
            end
            oldIdx=true(newSize,1);
            oldIdx(padIdx)=false;
            newThis.C(padIdx,:)=Cpad;
            newThis.C(oldIdx,:)=this.C;
            newThis.D(padIdx,:)=Dpad;
            newThis.D(oldIdx,:)=this.D;
            newThis.R(sub2ind([newSize, newSize],padIdx,padIdx))=Rpad;
            newThis.R(oldIdx,oldIdx)=this.R;
        end
        function newThis=reduce(this,excludeIdx)
           newThis=this;
           newThis.C(excludeIdx,:)=[];
           newThis.D(excludeIdx,:)=[];
           newThis.R(excludeIdx,:)=[];
           newThis.R(:,excludeIdx)=[];

        end
        function hs=get.hash(this)
           %This uses an external MEX function to compute the MD5 hash
           hs=GetMD5([this.A, this.B, this.Q, this.C'; this.C, this.D, this.C, this.R]);
        end
    end

    methods (Static)
        function [this,outlog]=id(datSet,order,opts) %randomStartEM wrapper for this class
            %To do: parallel process across datSets or orders
            if nargin<3
                opts.Nreps=0; %Simple EM, starting from PCA approximation
            end
            M=numel(order);
            if M>1 %Multiple orders to be fit
              if isa(datSet,'cell') && numel(datSet)>1 %Many datSets provided
                  N=numel(datSet);
              else
                if ~isa(datSet,'cell')
                  datSet={datSet};
                end
                N=1;
              end
                this=cell(M,N);
                outlog=cell(M,N);
                parfor ord=1:M %This allows for parallelism in model orders
                    for j=1:N
                      [this{ord,j},outlog{ord,j}]=linsys.id(datSet{j},order(ord),opts);
                    end
                end
            elseif isa(datSet,'cell') && numel(datSet)>1 %Single order, multi-set
                N=numel(datSet);
                this=cell(1,N);
                outlog=cell(1,N);
                parfor j=1:N %Parallelism in datasets
                  [this{j},outlog{j}]=linsys.id(datSet{j},order,opts);
                end
            else %Single order, single set
                if ~isfield(opts,'includeOutputIdx') || isempty(opts.includeOutputIdx)
                    ny=datSet.Noutput;
                elseif ~islogical(opts.includeOutputIdx)
                    ny=length(opts.includeOutputIdx);
                else
                    ny=sum(opts.includeOutputIdx);
                end
                if order==0
                    [J,B,C,D,Q,R,logL]=getFlatModel(datSet.out,datSet.in,opts);
                    this=fittedLinsys(J,C,R,B,D,Q,initCond([],[]),datSet,'EM',opts,logL,[]);
                    this.name='Flat';
                    outlog=[];
                    if isfield(opts,'fixR') && ~isempty(opts.fixR)
                      this.R=opts.fixR;
                    end
                elseif order>0
                    [A,B,C,D,Q,R,X,P,logL,outlog]=randomStartEM(datSet.out,datSet.in,order,opts);
                    iC=initCond(X(:,1),P(:,:,1)); %MLE init condition
                    %this=linsys(A,C,R,B,D,Q,trainInfo(datSet.hash,'repeatedEM',opts));
                    this=fittedLinsys(A,C,R,B,D,Q,iC,datSet,'repeatedEM',opts,logL,outlog);
                    this.name=['rEM ' num2str(order)];
                else
                    error('Order must be a non-negative integer.')
                end
            end
        end
        function this=struct2linsys(str)
            if iscell(str)
                for i=1:length(str(:))
                    this{i}=linsys.struct2linsys(str{i});
                end
                this=reshape(this,size(str));
            else
                if ~isfield(str,'A') && isfield(str,'J')
                    str.A=str.J;
                end
                this=linsys(str.A,str.C,str.R,str.B,str.D,str.Q);
            end
        end
        function fh=vizMany(modelCollection)
            mdl=cellfun(@(x) x.linsys2struct,modelCollection,'UniformOutput',false);
            vizModels(mdl)
        end
    end
end
