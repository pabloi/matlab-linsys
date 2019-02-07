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
    properties(Dependent)
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
        
        function [X,P,Xp,Pp,rejSamples] = Kfilter(this,datSet,initC,opts)
            if nargin<4
                opts=[];
            end
            if nargin<3
                initC=initCond([],[]);
            end
            [X,P,Xp,Pp,rejSamples]=statKalmanFilter(datSet.out,this.A,this.C,this.Q,this.R,initC.x,initC.P,this.B,this.D,datSet.in,opts);
        end
        function [X,P,Pt,Xf,Pf,Xp,Pp,rejSamples] = Ksmooth(this,datSet,initC,opts)
            if nargin<4
                opts=[];
            end
            if nargin<3
                initC=initCond([],[]);
            end
            [X,P,Pt,Xf,Pf,Xp,Pp,rejSamples]=statKalmanSmoother(datSet.out,this.A,this.C,this.Q,this.R,initC.x,initC.P,this.B,this.D,datSet.in,opts);
        end
        function fh=visualize(this)
            %TODO
            fh=figure;
        end
        function newThis=canonize(this,method)
            
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
    end
    
    methods(Static)
        function [this,outlog]=id(datSet,order,opts) %randomStartEM wrapper for this class
            if nargin<3
                opts.Nreps=0; %Simple EM, starting from PCA approximation
            end
            [A,B,C,D,Q,R,~,~,~,outlog]=randomStartEM(datSet.out,datSet.in,order,opts);
            this=linsys(A,C,R,B,D,Q,trainInfo(datSet.hash,'repeatedEM',opts));
        end
        function this=struct2linsys(str)
            if ~isfield(str,'A') && isfield(str,'J')
                str.A=str.J;
            end
            this=linsys(str.A,str.C,str.R,str.B,str.D,str.Q);
        end
    end
end

