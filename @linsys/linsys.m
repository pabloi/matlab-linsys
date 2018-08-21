classdef linsys
    %UNTITLED6 Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        A
        B
        C
        D
        Q
        R
        trainingOutput=[];
        trainingInput=[];
        trainingState=[];
        trainingStateUncertainty=[];
    end
    
    methods
        function this = linsys(A,C,R,B,D,Q) %Constructor
            this.A=A;
            this.C=C;
            this.R=R;
            this.B=B;
            this.D=D;
            this.Q=Q;
            %TODO: check sizes
        end
        
        function [X,P,Xp,Pp,rejSamples] = Kfilter(this,output,input,x0,P0,outlierRejection)
            if nargin<6
                outlierRejection=[];
            end
            if nargin<5
                P0=[];
                x0=[];
            end
            [X,P,Xp,Pp,rejSamples]=statKalmanFilter(output,this.A,this.C,this.Q,this.R,x0,P0,this.B,this.D,input,outlierRejection);
        end
        function [X,P,Pt,Xf,Pf,Xp,Pp,rejSamples] = Ksmooth(this,output,input,x0,P0,outlierRejection)
            if nargin<6
                outlierRejection=[];
            end
            if nargin<5
                P0=[];
                x0=[];
            end
            [X,P,Pt,Xf,Pf,Xp,Pp,rejSamples]=statKalmanSmoother(output,this.A,this.C,this.Q,this.R,x0,P0,this.B,this.D,input,outlierRejection);
        end
        function fh=visualize(this)
            %TODO
            fh=figure;
        end
        function newThis=canonize(this)
            
        end
        function newThis=upsample(this,Nfactor)
            warning('unimplemented')
           newThis=this; %Doxy 
        end
        function newThis=downsample(this,Nfactor)
            warning('unimplemented')
           newThis=this; %Doxy 
        end
        function fh=assess(this,output,input)
            %TODO
            fh=figure;
        end
        function l=logL(this,output,input)
            
            l=dataLogLikelihood(output,input,this.A,this.B,this.C,this.D,this.Q,this.R,x0,P0);
        end
    end
    
    methods(Static)
        function this=sysid(output,input,order)
            Nreps=10;
            method='true';
            [A,B,C,D,Q,R,X,P]=randomStartEM(output,input,order,Nreps,method);
            this=linsys(A,C,R,B,D,Q);
            this.trainingState=X;
            this.trainingStateUncertainty=P;
        end
    end
end

