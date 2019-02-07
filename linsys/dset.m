classdef dset
    %DSET is a class that defines input-output datasets for modeling. It
    %simply consists of two fields: the input and the output
    
    properties
        in
        out
    end
    properties(Dependent)
        Ninput %Number of inputs
        Noutput %Number of outputs
        Nsamp %Number of samples
        hash %MD5 hash
    end
    methods
        function this = dset(in,out)
            this.in=in;
            this.out=out;
            if size(in,2)~=size(out,2)
                error('dset:constructor','Inconsistent input and output sample sizes')
            end
        end
        function Nin=get.Ninput(this)
            Nin=size(this.Ninput,1);
        end
        function Nout=get.Noutput(this)
            Nout=size(this.Noutput,1);
        end
        function Nsample=get.Nsamp(this)
            Nsample=size(this.Ninput,2);
        end
        function hs=get.hash(this)
           %This uses an external MEX function to compute the MD5 hash
           hs=GetMD5([this.in;this.out]); 
        end
    end
end

