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
        isMultiple
    end
    methods
        function this = dset(in,out)
            if iscell(in) 
                this.in=in;
                if iscell(out) %Both are cells
                    this.out=out;
                else
                    for i=1:length(in)
                        this.out{i}=out;
                    end
                end
            elseif iscell(out) %Only out is cell
                this.out=out;
                for i=1:length(out)
                    this.in{i}=in;
                end
            else %Both are arrays
                this.in=in;
                this.out=out;
            end
            if ~this.isMultiple
                if size(in,2)~=size(out,2)
                    error('dset:constructor','Inconsistent input and output sample sizes')
                end
            else 
                if length(this.in)~=length(this.out)
                    error('dset:constructor','Inconsistent number of inputs and outputs');
                end
                for i=1:length(this.in)
                    if size(this.in{i},2)~=size(this.out{i},2)
                        error('dset:constructor','Inconsistent input and output sample sizes')
                    end
                end
            end
        end
        function flag=get.isMultiple(this)
            flag=iscell(this.in);
        end
        function Nin=get.Ninput(this)
            if ~this.isMultiple
                Nin=size(this.in,1);
            else
                Nin=cellfun(@(x) size(x,1),this.in);
            end
        end
        function Nout=get.Noutput(this)
            if ~this.isMultiple
                Nout=size(this.out,1);
            else
                Nout=cellfun(@(x) size(x,1),this.out);
            end
        end
        function Nsample=get.Nsamp(this)
            if ~this.isMultiple
                Nsample=size(this.in,2);
            else
                Nsample=cellfun(@(x) size(x,2),this.in);
            end
        end
        function N=get.nonNaNSamp(this)
            if ~this.isMultiple
                N=sum(~any(isnan(this.out))); %Non-NaN samples counted only
            else
                N=cellfun(@(x) sum(~any(isnan(x))),this.out);
            end
        end
        function hs=get.hash(this)
           %This uses an external MEX function to compute the MD5 hash
           if ~this.isMultiple
               hs=GetMD5([this.in;this.out]);
           else
               hs=GetMD5([cell2mat(this.in); cell2mat(this.out)]);
           end
        end
        function [res,resLS]=getDataProjections(this,model)
            if this.isMultiple
                error('Unimplemented')
            end
            yd=this.out-model.D*this.in;
            res=model.C\(yd);
            [CtRinvC,~,CtRinvY]=reduceModel(model.C,model.R,yd);
            resLS=CtRinvC\CtRinvY;
        end
        function newThis=reduce(this,excludeIdx)
            if this.isMultiple
                error('Unimplemented')
            end
            newThis=this;
            newThis.out(excludeIdx,:)=[];
        end
        function multiSet=split(this,breaks,returnAsMultiSet)
            if nargin<3
                returnAsMultiSet=false;
            end
            if this.isMultiple
                error('Unimplemented')
            end
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
            if returnAsMultiSet
                multiSet=dset(newIn,newOut);
            else
            multiSet=cell(N,1);
            for i=1:N
              multiSet{i}=dset(newIn{i},newOut{i});
            end
            end
        end
        function multiSet=blockSplit(this,blockSize,Npartitions)
            if this.isMultiple
                error('Unimplemented')
            end
            %Splits the dataset into Npartitions by alternating blocks of blockSize
            %If number of samples is not an exact multiple of blocksize, the last incomplete block is discarded and not assigned anywher.
            %There is no guarantee that the partitions will have equal sizes
            Nblocks=floor(this.Nsamp/blockSize);
            lastSample=Nblocks*blockSize;
            this.out(:,lastSample+1:end)=NaN; %Deleting unused samples for everyone
            multiSet=cell(Npartitions,1);
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
            for j=1:Npartitions %Go through partitions and remove leading/trailing NaNs to avoid ill-conditioned fitting
               nanSamples=all(isnan(multiSet{j}.out));
               firstNonNaN=find(~nanSamples,1,'first');
               lastNonNaN=find(~nanSamples,1,'last');
               multiSet{j}.out=multiSet{j}.out(:,firstNonNaN:lastNonNaN);
               multiSet{j}.in=multiSet{j}.in(:,firstNonNaN:lastNonNaN);
            end
        end
        function multiSet=alternate(this,N)
            if this.isMultiple
                error('Unimplemented')
            end
           %Creates N different data folds by putting 1 every N datapoints
           %into each dataset
           [ou] = foldSplit(this.out',N); %Foldsplit works along first dim
           multiSet=cell(size(ou));
           for i=1:N
                multiSet{i}=dset(this.in,ou{i}');
           end
        end
        function [fh,fh2]=vizFit(this,models)
            if this.isMultiple
                error('Unimplemented')
            end
            [fh,fh2] = vizDataFit(models,this);
        end
        function fh=vizRes(this,models)
            if this.isMultiple
                error('Unimplemented')
            end
            [fh] = vizDataRes(models,this);
        end
        function fh=compareModels(this,models)
            if this.isMultiple
                error('Unimplemented')
            end
            %To do: if model is fittedLinsys and the hash of all the
            %datasets coincides with the hash of this, use
            %fittedLinsys.compare()
            [fh] = vizDataLikelihood(models,this);
        end
        function l=logL(this,mdl,initC)
            if this.isMultiple
                error('Unimplemented')
            end
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
        function r=flatResiduals(this)
            if this.isMultiple
                error('Unimplemented')
            end
            [J,B,C,D,Q,R,logLperSamplePerDim]=getFlatModel(this.out,this.in);
            r=this.out-D*this.in;
        end
        function W=estimateVar(this)
            if this.isMultiple
                error('Unimplemented')
            end
           diffU=diff(this.in,[],2);
           diffY=diff(this.out,[],2);
           diffY=diffY(:,all(diffU==0)); %sample differences when input=constant
           W=.5*(diffY*diffY')/size(diffY,2); %Covariance of said samples
           %Formally, if data comes from a linear system:
           %E(W)=R+.5*CQC'+.5*H*(x-x_inf)*(x-x_inf)'*H'
           %Where H=C*(A-I), x_inf=steady-state for the given output at
           %that time. If most samples are close to steady-state, then the
           %third term is negligible, and this becomes an estimate of
           %R+.5*CQC'
        end
        function newThis=extractSingle(this,i)
           if this.isMultiple
               if i>length(this.out)
                   error('dset:extractSingle',['Single index provided (' num2str(i) ') is larger than available number of dsets in this object (' num2str(length(this.out)) ').'])
               else
                   newThis=dset(this.in{i},this.out{i});
               end
           else
               error('dset object is not multiple, cannot extract a single set')
           end
        end
    end
end
