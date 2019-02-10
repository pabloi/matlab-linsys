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
        %Optional: training info, could contain anything
        trainingInfo=trainInfo();
        name='';
    end
    properties (Dependent)
        order %Model order, number of stats
        Ninput %Number of inputs
        Noutput %Number of outputs
    end

    methods
        function this = linsys(A,C,R,B,D,Q,trainInfo) %Constructor
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
            if nargin>6
                this.trainingInfo=trainInfo;
            end
        end
        function dfit=fit(this,datSet,initC)
            if nargin<3
                initC=[];
            end
            dfit=dataFit(this,datSet,initC);
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
                [X,P,Xp,Pp,rejSamples,logL]=statKalmanFilter(datSet.out,this.A,this.C,this.Q,this.R,initC.x,initC.P,this.B,this.D,datSet.in,opts);
                filteredState=stateEstimate(X,P);
                oneAheadState=stateEstimate(Xp,Pp);
                logL=logL*numel(datSet.out);%logL from Ksmooth is given on a per-sample per-dim basis;
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
                if nargin<3 || isempty(initC)
                    initC=initCond(zeros(this.order,0)); %Improper prior
                end
                [X,P,Pt,Xf,Pf,Xp,Pp,rejSamples,logL]=statKalmanSmoother(datSet.out,this.A,this.C,this.Q,this.R,initC.state,initC.covar,this.B,this.D,datSet.in,opts);
                smoothState=stateEstimate(X,P,Pt);
                filteredState=stateEstimate(Xf,Pf);
                oneAheadState=stateEstimate(Xp,Pp);
                logL=logL*numel(datSet.out);%logL from Ksmooth is given on a per-sample per-dim basis;
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
        function [datSet,stateE]=simulate(this,input,initC,deterministicFlag)
            %Simulates a realization of the modeled system given the initial
            %conditions (uncertainty in inital condition is ignored)
            %If deterministicFlag=true, then Q is set to 0 before simulating
            %(there is still observational noise, but states evolve in a deterministic way)
            if nargin>3 && deterministicFlag
                this.Q=zeros(size(this.Q));
            end
            if nargin<3
                iC=[];
            else
                iC=initC.state;
            end
            [out,state]=fwdSim(input,this.A,this.B,this.C,this.D,iC,this.Q,this.R);
            datSet=dset(input,out);
            stateE=stateEstimate(state,zeros(this.order,this.order,datSet.Nsamp));
        end
        function fh=visualize(this)
            %TODO
            fh=figure;
        end
        function [this,V]=canonize(this,method)
          if nargin<2
            method=[];
          end
          [this.A,this.B,this.C,~,V,this.Q,~] = canonize(this.A,this.B,this.C,[],this.Q,[],method);
        end
        function newThis=upsample(this,Nfactor)
            error('unimplemented')
           newThis=this; %Doxy
        end
        function newThis=downsample(this,Nfactor)
            error('unimplemented')
           newThis=this; %Doxy
        end
        function fh=assess(this,datSet,initC)
            %TODO
            fh=figure;
        end
        function l=logL(this,datSet,initC)
            if nargin<3
                initC=initCond([],[]);
            end
            l=dataLogLikelihood(datSet.out,datSet.in,this.A,this.B,this.C,this.D,this.Q,this.R,initC.x,initC.P);
        end
        function ord=get.order(this)
            ord=size(this.A,1);
        end
        function df=dof(this)
            %Computes effective degrees of freedom of the system, assuming all non-zero parameters were freely selected.
            %Warning: this presumes a diagonal A matrix, and counting non-zero entries post-diagonalization. This is not necessarily the way that results in the least amount of non-zero parameters (can this be proved?).
            this=this.canonize; %Diagonalizing
            Na=this.order; %Using diagonal A.
            Nb=sum(sum(this.B~=0));
            Nc=sum(sum(this.C~=0));
            if all(this.C==1); Nc=0; end; %The 1 value is assigned for flat models, this is an ugly workaround
            Nd=sum(sum(this.D~=0));
            Nq=sum(sum(triu(this.Q)~=0));
            Nr=sum(sum(triu(this.R)~=0));
            Nredundant= this.order; %Up to this.order parameters can be arbitrarily set. For example, arbitrarily scaling all states and C accordingly.
            df=Na+Nb+Nc+Nd+Nq+Nr-Nredundant; %Model free parameters
        end
    end

    methods (Static)
        function [this,outlog]=id(datSet,order,opts) %randomStartEM wrapper for this class
            if nargin<3
                opts.Nreps=0; %Simple EM, starting from PCA approximation
            end
            if numel(order)>1
                M=length(order);
                this=cell(M,1);
                outlog=cell(M,1);
                for ord=1:M
                    [this{ord},outlog{ord}]=linsys.id(datSet,order(ord),opts);
                end
            else
                if order==0
                    [J,B,C,D,Q,R]=getFlatModel(datSet.out,datSet.in);
                    this=linsys(J,C,R,B,D,Q,trainInfo(datSet.hash,'flatModel',[]));
                    this.name='Flat';
                    outlog=[];
                elseif order>0
                    [A,B,C,D,Q,R,~,~,~,outlog]=randomStartEM(datSet.out,datSet.in,order,opts);
                    this=linsys(A,C,R,B,D,Q,trainInfo(datSet.hash,'repeatedEM',opts));
                    this.name=['rEM ' num2str(order)];
                else
                    error('Order must be a non-negative integer.')
                end
            end
        end
        function this=struct2linsys(str)
            if ~isfield(str,'A') && isfield(str,'J')
                str.A=str.J;
            end
            this=linsys(str.A,str.C,str.R,str.B,str.D,str.Q);
        end
    end
end
