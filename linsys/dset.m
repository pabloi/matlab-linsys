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
           hs=GetMD5([this.in;this.out]);
        end
        function [res,resLS]=getDataProjections(this,model)
            yd=this.out-model.D*this.in;
            res=model.C\(yd);
            [CtRinvC,~,CtRinvY]=reduceModel(model.C,model.R,yd);
            resLS=CtRinvC\CtRinvY;
        end
        function newThis=reduce(this,excludeIdx)
            newThis=this;
            newThis.out(excludeIdx,:)=[];
        end
        function multiSet=split(this,breaks)
            %Splits a dataset along the specified breaks. Returns a cell-array of dset.
            %Breaks is a vector indicating the first data sample of each sub-set.
            %New sets are contiguous (previous one ends on the last sample before current one).
            %First set is presumed to start at 1, and last set is presumed to finish at end.
            if breaks(1)~=1
              breaks=[1; breaks(:)];
            end
            if breaks(end)~=this.Nsamp+1
              breaks=[breaks(:); this.Nsamp+1];
            end
            newIn=mat2cell(this.in,this.Ninput,diff(breaks));
            newOut=mat2cell(this.out,this.Noutput,diff(breaks));
            N=length(newIn);
            multiSet=cell(N,1);
            for i=1:N
              multiSet{i}=dset(newIn{i},newOut{i});
            end
        end
        function multiSet=blockSplit(this,blockSize,Npartitions)
            %Splits the dataset into Npartitions by alternating blocks of blockSize
            %If number of samples is not an exact multiple of blocksize, the last incomplete block is discarded and not assigned anywher.
            %There is no guarantee that the partitions will have equal sizes
            Nblocks=floor(this.Nsamp/blockSize);
            lastSample=Nblocks*blockSize;
            this.out(:,lastSample+1:end)=NaN; %Deleting unused samples for everyone
            for j=1:Npartitions
              multiSet{j}=this;
            end
            for i=1:Nblocks %For each block
              blockBegin=(i-1)*blockSize+1;
              blockEnd=i*blockSize;
              for j=1:Npartitions %Go through partitions
                if (mod(i-1,Npartitions)+1)~=j %This happens for all partitions except 1, alternating
                  multiSet{j}.out(:,blockBegin:blockEnd)=NaN; %Delete data for those partitions
                end
              end
            end
        end
        function multiSet=alternate(this,N)
           %Creates N different data folds by putting 1 every N datapoints
           %into each dataset
           [ou] = foldSplit(this.out',N); %Foldsplit works along first dim
           multiSet=cell(size(ou));
           for i=1:N
                multiSet{i}=dset(this.in,ou{i}');
           end
        end
        function [fh,fh2]=vizFit(this,models)
            [fh,fh2] = vizDataFit(models,this);
        end
        function fh=vizRes(this,models)
            [fh] = vizDataRes(models,this);
        end
        function fh=compareModels(this,models)
            %To do: if model is fittedLinsys and the hash of all the
            %datasets coincides with the hash of this, use
            %fittedLinsys.compare()
            [fh] = vizDataLikelihood(models,this);
        end
        function l=logL(this,mdl,initC)
            if nargin<3
                initC=initCond([],[]);
            end
            if isa(mdl,'cell')
              for i=1:numel(mdl)
                l(i)=this.logL(mdl{i},initC); %Recursive call to this func
              end
            else
              l=mdl.logL(this);
            end
        end
    end
end
