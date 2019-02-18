classdef dset
    %DSET is a class that defines input-output datasets for modeling. It
    %simply consists of two fields: the input and the output

    properties
        in
        out
    end
    properties (Dependent)
        Ninput %Number of inputs
        Noutput %Number of outputs
        Nsamp %Number of samples
        hash %MD5 hash
        nonNaNSamp
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
            Nin=size(this.in,1);
        end
        function Nout=get.Noutput(this)
            Nout=size(this.out,1);
        end
        function Nsample=get.Nsamp(this)
            Nsample=size(this.in,2);
        end
        function N=get.nonNaNSamp(this)
            N=sum(~any(isnan(this.out))); %Non-NaN samples counted only
        end
        function hs=get.hash(this)
           %This uses an external MEX function to compute the MD5 hash
           hs=dset.GetMD5([this.in;this.out]);
        end
        function multiSet=split(this,breaks)
            %Splits a dataset along the specified breaks. Returns a cell-array of dset.
            %Breaks is a vector indicating the first data sample of each sub-set.
            %New sets are contiguous (previous one ends on the last sample before current one).
            %First set is presumed to start at 1, and last set is presumed to finish at end.
            if breaks(1)~=1
              breaks=[1 breaks(:)];
            end
            if breaks(end)~=this.Nsamp+1
              breaks=[breaks(:) this.Nsamp+1];
            end
            newIn=mat2cell(this.in,this.Ninput,diff(breaks));
            newOut=mat2cell(this.out,this.Noutput,diff(breaks));
            for i=1:length(newIn)
              multiSet{i}=dset(newIn{i},newOut{i});
            end
        end
    end
    methods(Hidden,Static)
        function H = GetMD5(Data)
            %Modified from: https://www.mathworks.com/matlabcentral/fileexchange/31272-datahash
            %Copyright (c) 2018, Jan Simon
            %All rights reserved.

            %Redistribution and use in source and binary forms, with or without
            %modification, are permitted provided that the following conditions are met:
            %
            %* Redistributions of source code must retain the above copyright notice, this
            %  list of conditions and the following disclaimer.
            %
            %* Redistributions in binary form must reproduce the above copyright notice,
            %  this list of conditions and the following disclaimer in the documentation
            %  and/or other materials provided with the distribution
            %* Neither the name of University Heidelberg nor the names of its
            %  contributors may be used to endorse or promote products derived from this
            %  software without specific prior written permission.
            %THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
            %AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
            %IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
            %DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
            %FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
            %DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
            %SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
            %CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
            %OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
            %OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
            Engine = java.security.MessageDigest.getInstance('MD5');
            H = double(typecast(Engine.digest, 'uint8'));
            Engine.update(typecast(Data(:), 'uint8'));
            H = bitxor(H, double(typecast(Engine.digest, 'uint8')));
            H = sprintf('%.2x', H);   % To hex string
        end
    end
end
