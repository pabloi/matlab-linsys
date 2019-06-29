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
                    [filteredState{i},oneAheadState{i},rejSamples{i},logL(i)] = Kfilter(this,datSet{i},initC{i},opts);
                end
            elseif datSet.isMultiple
                for i=1:length(datSet.out)
                    if nargin<3 || isempty(initC)
                        iC=initCond([],[]);
                    else
                        iC=initC.extractSingle(i);
                    end
                    [filteredState{i},oneAheadState{i},rejSamples{i},logL(i)] = Kfilter(this,datSet.extractSingle(i),iC,opts);
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
            elseif datSet.isMultiple
                for i=1:length(datSet.out)
                    if nargin<3 || isempty(initC)
                        iC=initCond([],[]);
                    else
                        iC=initC.extractSingle(i);
                    end
                    [smoothState{i},filteredState{i},oneAheadState{i},rejSamples{i},logL(i)] = Ksmooth(this,datSet.extractSingle(i),iC,opts);
                end
            else
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
            %N is given by the input size
            %All states are assumed to evolve with the same input
            if nargin<3 %Projecting a single stride in the future, with null input
                in=zeros(this.Ninput,1);
            end
            x=zeros(size(stateE.state));
            P=zeros(size(stateE.covar));
            for j=1:stateE.Nsamp
                x(:,j)=stateE.state(:,j);
                P(:,:,j)=stateE.covar(:,:,j);
                for i=1:size(in,2)
                    x(:,j)=this.A*x(:,j) + this.B*in(:,i); %Project a single step into the future
                    P(:,:,j)=this.A*P(:,:,j)*this.A' + this.Q;
                end
            end
            stateE=stateEstimate(x,P);
        end
        function stateE=predict2(this,stateE,in,M)
            %Predicts new states for N samples in the future, given a
            %state series (assmued consecutive, each point is predicted M samples into the future using a single input series)
            %Input samples are assumed to be temporally aligned with the
            %stateE samples.
            x=zeros(size(stateE.state));
            P=zeros(size(stateE.covar));
            for j=1:(size(in,2)-M)
                x(:,j)=stateE.state(:,j);
                P(:,:,j)=stateE.covar(:,:,j);
                for i=1:M
                    x(:,j)=this.A*x(:,j) + this.B*in(:,i+j); %Project a single step into the future
                    P(:,:,j)=this.A*P(:,:,j)*this.A' + this.Q;
                end
            end
            stateE=stateEstimate(x,P);
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
        function this=scale(this,k)
           %Transforms the system by scaling states
           if numel(k)==1 %All states scaled equally
               k=k*ones(this.order,1);
           elseif numel(k)~=this.order
               error('')
           end
           this=this.transform(diag(k));
        end
        function [this]=transform(this,V)
            [this.A,this.B,this.C,this.Q]=transform(V,this.A,this.B,this.C,this.Q);
        end
        function this=shiftStates(this,K)
            %Transforms the model by trying to accomodate new states x'=x+K*u 
            I=eye(size(this.A));
            this.B=this.B+(I-this.A)*K;
            this.D=this.D-this.C*K;
        end
        function [this,K]=mleShift(this,datSet)
           %Finds the MLE shift of the model to accomodate a dataSet (as in shiftStates) 
           N=this.order;
           M=this.Ninput;
           zeroInds=all(this.B==0); %Ignoring these
           M=M-sum(zeroInds);
           K0=zeros(N,M);
           k=fminunc(@(x) -this.shiftStates(reshape(x,N,M)).logL(datSet),K0(:));
           aux=reshape(k,N,M);
           K=zeros(size(this.B));
           K(:,~zeroInds)=aux;
           this=shiftStates(this,K);
        end
        function this=EMrefine(this,datSet)
           %Runs the EM algorithm to maximize logL of the parameters for the dataset given 
           %with this model as starting point 
           smoothState=this.Ksmooth(datSet);
           opts.indB=~all(this.B==0); %Respecting the mask of B
           opts.indD=~all(this.D==0); %Respecting the mask of D
           opts.Niter=1000; %To avoid too long refinement
           opts.convergenceTol=1e-5;
           if isa(this,'fittedLinsys')
           opts.includeOutputIdx=this.trainOptions.includeOutputIdx; %Only fitting to subset of data, as originally done
           end
           if isa(smoothState,'cell')
               st=cellfun(@(x) x.state,smoothState,'UniformOutput',false);
               P=cellfun(@(x) x.covar,smoothState,'UniformOutput',false);
               Pt=cellfun(@(x) x.lagOneCovar,smoothState,'UniformOutput',false);
           else
               st=smoothState.state;
               P=smoothState.covar;
               Pt=smoothState.lagOneCovar;
           end
           [this.A,this.B,this.C,this.D,this.Q,this.R]=EM(datSet.out,datSet.in,st,opts,P,Pt);
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
        function [f1]=vizSingleFit(this,datSet,initC)
            if nargin<3
                initC=[];
            end
           dfit=this.fit(datSet,initC,'KS'); %MLE fit
           f1=figure('Name','State fits','Units','Pixels','InnerPosition',[100 100 300*6 300*3]);
           %For each state and input associated with NON-ZERO B column,
           %plot state estimates, data projections, and contribution to
           %output
           Mx=this.order;
           indB=find(sum(this.B~=0)~=0);
           Mu=length(indB);
           M=Mx+Mu;
           xMargin=.05;
           xWidth=.9/M;
           xCoverage=.9; %Determines whitespace
           yMargin=.05;
           yHeight=(1-2*yMargin)/3;
           yCoverage=.9;
           CD=[this.C this.D];
           dataProj=(CD\datSet.out);
           mC=max(max(abs(this.D(:,indB))));
           %States first:
           for i=1:M
               %States:
               ax=axes('Position',[xMargin+xWidth*(i-1) yMargin+2*yHeight xCoverage*xWidth yCoverage*yHeight]);
               scatter(1:size(dataProj,2),dataProj(i,:),5,.5*ones(1,3),'filled','MarkerFaceAlpha',.3,'DisplayName','Data projection')
               hold on
               if i<=Mx
                    dfit.stateEstim.marginalize(i).plot(0,[],ax); %States
                    addedTXT=[', \tau = ' num2str(-1./log(this.A(i,i)),3) ', b = ' num2str(this.B(i,indB),2) ];
                    title(['State ' num2str(i) addedTXT])
                    pp=findobj(ax,'Type','Patch');
                    pp.DisplayName='99.7% CI';
                    pp=findobj(ax,'Type','Line');
                    pp.DisplayName='State MLE';
                    uistack(pp,'top')
               else
                   pp=plot(datSet.in(indB(i-Mx),:),'LineWidth',2,'DisplayName','Input');
                   uistack(pp,'top')
                   if Mu>1
                    title(['Input ' num2str(indB(i-Mx))])
                   else
                       title('Input')
                   end
               end
               axis tight
               ax.XAxis.Limits=[1,datSet.Nsamp];
               if i==1
                   legend('Location','NorthEast','Box','off')
               end
               %C,D columns
               ax=axes('Position',[xMargin+xWidth*(i-1) yMargin xCoverage*xWidth yHeight+yCoverage*yHeight]);
               imagesc(reshape(CD(:,i),12,15)')
               ex1=[0.8500    0.3250    0.0980]; %2nd MAtlab default color
               ex2=[0.4660    0.6740    0.1880]; %5th Matlab default color
               gamma=1;
               map=[bsxfun(@plus,ex1.^(1/gamma),bsxfun(@times,1-ex1.^(1/gamma),[0:.01:1]'));bsxfun(@plus,ex2.^(1/gamma),bsxfun(@times,1-ex2.^(1/gamma),[1:-.01:0]'))].^gamma;
               colormap(map)
               caxis([-1 1]*mC)
               if i==M
                   pos=ax.Position;
                   colorbar
                   ax.Position=pos;
               end
           end
        end
        function f2=vizSingleRes(this,datSet,initC,windows,Nahead)
            if nargin<5 || isempty(Nahead)
                Nahead=1;
            end
            if nargin<3
                initC=[];
            end
            if nargin<4 || isempty(windows)
                windows=sort(round(datSet.Nsamp*rand(5,1)));
            end
            winSize=5;
            windows(windows>datSet.Nsamp-winSize)=[];
            windows(windows<1)=[];
            dfit=this.fit(datSet,initC,'KF'); %MLE fit  
            predictedOut=dfit.NaheadOutput(Nahead);
            predictedRes=datSet.out-predictedOut;
           f2=figure('Name','Data residuals','Units','Pixels','InnerPosition',[100 100 300*4.1 300*4]); 
           M=length(windows);
           xMargin=.05;
           xWidth=.86/M;
           xCoverage=.9; %Determines whitespace
           yMargin=.05;
           yHeight=(1-2*yMargin)/5;
           yCoverage=.9;
           indB=find(sum(this.B~=0)~=0);
           mC=max(max(abs(this.D(:,indB)))); %Same scaling as vizSingleFit
           ex1=[0.8500    0.3250    0.0980]; %2nd MAtlab default color
           ex2=[0.4660    0.6740    0.1880]; %5th Matlab default color
           gamma=1;
           map=[bsxfun(@plus,ex1.^(1/gamma),bsxfun(@times,1-ex1.^(1/gamma),[0:.01:1]'));bsxfun(@plus,ex2.^(1/gamma),bsxfun(@times,1-ex2.^(1/gamma),[1:-.01:0]'))].^gamma;
                 
           for i=1:M
               relevantSamples=windows(i)+[1:winSize]-1;
               for j=1:3 %Data, prediction, residual images
                    ax=axes('Position',[xMargin+(i-1)*xWidth yMargin+(4-(j-1))*yHeight xCoverage*xWidth yCoverage*yHeight]);
                    switch j
                        case 1 %Data
                            d=mean(datSet.out(:,relevantSamples),2);
                            ttl={ 't=' ; [ '[' num2str(windows(i)) ',' num2str(windows(i)+winSize) ']' ] };
                            yl='data';
                        case 2 %Prediction
                            d=mean(predictedOut(:,relevantSamples),2);
                            yl={[num2str(Nahead) '-ahead'];['prediction']};
                        case 3 %Residuals
                            d=mean(predictedRes(:,relevantSamples),2);
                           yl={[num2str(Nahead) '-ahead'];['residual']};
                    end
                    imagesc(reshape(d,12,15)')
                     colormap(map)
                   caxis([-1 1]*mC)
                   if i==M && j==1
                       pos=ax.Position;
                       colorbar
                       ax.Position=pos;
                   end
                   if j==1
                       title(ttl)
                   end
                   if i==1
                   ylabel(yl)
                   end
                   ax.XAxis.TickValues=[];
                   ax.YAxis.TickValues=[];
               end
           end
           % Residual time courses: (RMSE)
           ax=axes('Position',[xMargin yMargin+(1)*yHeight xCoverage*xWidth+(M-2)*xWidth yCoverage*yHeight]);
           rmse=sqrt(sum(predictedRes.^2));
           rmsePrevSample=[nan sqrt(sum((datSet.out(:,2:end)-datSet.out(:,1:end-1)).^2))];
           avg11=conv2(datSet.out,ones(1,11)/11,'same');
           avgPrev11=[nan(size(datSet.out,1),11) avg11(:,6:end-6)];
           rmsePrev11= sqrt(sum((datSet.out-avgPrev11).^2));
           rmseFlat=sqrt(sum((datSet.out-datSet.out/datSet.in *datSet.in).^2));
           (rmseFlat/rmse)^2
           filtSize=3;
           rmse=medfilt1(rmse,filtSize,'truncate');
           rmsePrevSample=medfilt1(rmsePrevSample,filtSize,'truncate');
           rmsePrev11=medfilt1(rmsePrev11,filtSize,'truncate');
           rmseFlat=medfilt1(rmseFlat,filtSize,'truncate');
           hold on
           plot(rmsePrevSample,'k','DisplayName','Prev. sample')
           %plot(rmsePrev11,'Color',.5*ones(1,3),'DisplayName', 'Prev. 11 samples')
           plot(rmseFlat,'Color',.5*ones(1,3),'DisplayName', 'Flat model')
           set(ax,'ColorOrderIndex',1)
           plot(rmse,'LineWidth',2,'DisplayName','3-state model')
           ax.YAxis.Scale='log';
           ylabel({'RMSE';'residual'})
           ax.XAxis.Limits=[1 length(rmse)];
           ax.XAxis.TickValues=[];
           legend('Location','NorthEast')
           %ax=axes('Position',[xMargin+(M-1)*xWidth yMargin+(1)*yHeight xCoverage*xWidth yCoverage*yHeight]);
               
           % First PC of residual, timecourse and image:
           nanIdx=any(isnan(predictedRes));
           [p,c,a]=pca(predictedRes(:,~nanIdx),'Centered','off');
           k=sqrt(sum(c(:,1).^2))/sqrt(sum(this.D(:,1).^2));
           c=c/k;
           p=p*k;
           v=a(1)/sum(a); %Variance explained by first PC
           ax=axes('Position',[xMargin yMargin xCoverage*xWidth+(M-2)*xWidth yCoverage*yHeight]);
           %plot(find(~nanIdx),p(:,1),'LineWidth',2)
           scatter(find(~nanIdx),p(:,1),5,.4*ones(1,3),'filled','MarkerEdgeColor','none')
           ylabel({'PC1';' of residual'})
           ax.XAxis.Limits=[1 length(rmse)];
           ax=axes('Position',[xMargin+(M-1)*xWidth yMargin xCoverage*xWidth 2*yCoverage*yHeight]);
           imagesc(reshape(c(:,1),12,15)')
           colormap(map)
           caxis([-1 1]*mC)
           title(['PC1 of residual (' num2str(v*100,2) '%)'])
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
        end
        function r=residual(this,datSet,method,iC)
            if nargin<3
                method='det';
            end
            if nargin<4 %No initial condition provided, computing an appropriate one
                if strcmp(method,'oneAhead') %For one ahead predictions, could also use infinite uncertainty as initCond
                    %USING MLE initial Condition
                    dfit=this.fit(datSet,[],'KS');
                elseif strcmp(method,'det')
                    %IF the method is deterministic, it is best to look for
                    %an initial condition  forcing Q=0 (it will be a more
                    %likely value for a deterministic system).
                    this.Q=zeros(size(this.Q));
                    dfit=this.fit(datSet,[],'KF'); %Using KF because KS does not work well with Q=0 (backpropagation issues, need to review)
                end
                iC=dfit.stateEstim.getSample(1);
            end
            dfit=this.fit(datSet,iC,'KF');
            switch method
                case 'oneAhead'
                    r=dfit.oneAheadResidual;
                case 'det'
                    r=dfit.deterministicResidual;
            end
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
            if nargin>1
            X=(I-A)\(I-A^N)*B(:,1);
            else %Infinite time-horizon, presuming stable system
                X=(I-A)\B(:,1);
            end
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
        function newThis=excludeOutput(this,excludeIdx)
           newThis=this;
           newThis.C(excludeIdx,:)=[];
           newThis.D(excludeIdx,:)=[];
           newThis.R(excludeIdx,:)=[];
           newThis.R(:,excludeIdx)=[];
        end
        function [newThis,redDataSet]=reduce(this,datSet)
            %dataSet argument is optional
            if nargin<2
                Y=zeros(this.Noutput,1);
            else
                Y=datSet.out;
            end
            exc=isinf(diag(this.R));
            this=this.excludeOutput(exc); %Excluding infinite variance outputs
            Y=Y(~exc,:);
           [Cnew,Rnew,Ynew,cRnew,logLmargin,Dnew]=reduceModel(this.C,this.R,Y,this.D);
           newThis=linsys(this.A,Cnew,Rnew,this.B,Dnew,this.Q);
           redDataSet=dset(datSet.in,Ynew);
        end
        function hs=get.hash(this)
           %This uses an external MEX function to compute the MD5 hash
           hs=GetMD5([this.A, this.B, this.Q, this.C'; this.C, this.D, this.C, this.R]);
        end
        function compTbl=comparisonTable(this,altModels)
            %compTbl=table(,'VariableNames',{'\tau','B_1','diag(Q)','tr(R)'}) %'norm(D-\hat{D})_F','norm(R-\hat{R})_F'},
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
                    if ~datSet.isMultiple
                        iC=initCond(X(:,1),P(:,:,1)); %MLE init condition
                    else
                        iC=initCond(cellfun(@(x) x(:,1),X,'UniformOutput',false),cellfun(@(x) x(:,:,1),P,'UniformOutput',false));
                    end
                    %this=linsys(A,C,R,B,D,Q,trainInfo(datSet.hash,'repeatedEM',opts));
                    this=fittedLinsys(A,C,R,B,D,Q,iC,datSet,'repeatedEM',opts,logL,outlog);
                    this.name=['rEM ' num2str(order)];
                else
                    error('Order must be a non-negative integer.')
                end
            end
        end
        function [this,outlog]=SSid(datSet,order,ssSize,method)
            %Model identification through subspace model
            if nargin<4
                method='SS';
            end
            if nargin<3
                ssSize=10; %Default value
                if nargin<2
                    order=[];
                end
            end
            switch method
                case 'SS' %Fast-ish implementation, as described in 
                    %Shadmehr and Mussa-Ivaldi 2012, 
                    %and van Overschee and de Moor 1996 (Chap. 4, Algo. 2)
                    %As implemented by me.
                    [J,B,C,D,~,Q,R]=subspaceID(datSet.out,datSet.in,order,ssSize);
                case 'SSunb' %Unbiased version,
                     %van Overschee and de Moor 1996 (Chap. 4, Algo. 1 with some suggested improvements from Algo. 3)
                     %As implemented by me.
                    [J,B,C,D,~,Q,R]=subspaceIDunbiased(datSet.out,datSet.in,order,ssSize);
                case 'SSEM' %A hybrid SS-EM implementation. J is estimated from SSunb, but all other parameters from EM
                    [J,B,C,D,~,Q,R]=subspaceEMhybrid(datSet.out,datSet.in,order,ssSize);
                case 'subid' %van Overschee's own implementation of Chap. 4, Algo. 3
                    %Very fast, although for some reason requires more data
                    %for same value of ssSize
                    [J,B,C,D,K,R] = subid(datSet.out,datSet.in,ssSize,order);
                    cR=chol(R);
                    KcQ=K*cR;
                    Q=KcQ*KcQ';
            end
           mdl=linsys(J,C,R,B,D,Q);
           this=fittedLinsys(J,C,R,B,D,Q,initCond([],[]),datSet,[method '_i' num2str(ssSize)],[],mdl.logL(datSet),[]);
           outlog=[];
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
        function compTbl=summaryTable(models)
            mdl=cellfun(@(x) x.canonize('canonicalAlt').scale(1/sqrt(sum(x.D(:,1).^2))),models,'UniformOutput',false);
            M=numel(mdl);
            N=max(cellfun(@(x) size(x.A,1),mdl));
            %Check: all models should be same order(?)
            taus=nan(N,M);
            B1=nan(N,M);
            dQ=nan(N,M);
            trR=nan(M,1);
            minR=nan(M,1);
            maxR=nan(M,1);
            for i=1:numel(mdl) %Each model
               aux= sort(-1./log(eig(mdl{i}.A)));
               taus(1:length(aux),i) =aux;
               aux=mdl{i}.B(:,1);
               Bees(1:length(aux),i)=aux;
               A=mdl{i}.A;
               aux=(eye(size(A))-A)\aux; %steady-state x under step input in first input component
               xinf(1:length(aux),i)=aux;
               aux=mdl{i}.detPredict;
               B1(1:length(aux),i)=aux;
               dR=diag(mdl{i}.R);
               dR=dR(~isinf(dR)); %Removing infinite values
               trR(i)=sum(dR);
               minR(i)=min(dR);
               maxR(i)=max(dR);
               aux=diag(mdl{i}.Q);
               dQ(1:length(aux),i)=aux;
            end
            varNames={};
            varTbl=[];
            for i=1:N
                aux={['T_' num2str(i)],['Q_' num2str(i)],['B_' num2str(i)]};
                varNames=[varNames aux];
                varTbl=[varTbl taus(i,:)' dQ(i,:)' Bees(i,:)'];
            end
            varTbl=[varTbl trR minR maxR];
            varNames=[varNames {'trR','minR','maxR'}];
            compTbl=array2table(varTbl);
            compTbl.Properties.VariableNames=varNames;
            compTbl.Properties.RowNames=cellfun(@(x) x.name, mdl, 'UniformOutput',false);
        end
        function fh=compareResiduals(modelCollection,datSet,method)
           if ~isa(datSet,'cell')
               datSet={datSet};
           end
           N=length(datSet);
           fh=figure;
           if nargin<3 || isempty(method)
               method='det';
           end
           for j=1:N %Each dataset
               M=length(modelCollection);
               r=nan(M,1);
               for i=1:M %Each model
                   res=modelCollection{i}.residual(datSet{j},method);
                   res=sum(res.^2);
                   if any(isnan(res))
                       warning('Found NaN residuals, this may be caused by filtering from improper priors. Proceeding by ignoring first NaN samples')
                       firstNonNaN=find(~isnan(res),1,'first');
                       res=res(firstNonNaN:end);
                   end
                   r(i)=sqrt(sum(res)); %RMSE
               end
               subplot(1,N,j)
               cc=get(gca,'ColorOrder');
               r=r/r(1); %Normalizing to first model's residuals
               bar(100*[1:length(r)],r,'FaceColor',cc(1,:),'EdgeColor','w','FaceAlpha',.5,'BarWidth',1)
               ax=gca;
               ax.YAxis.Limits=[min(r),1.05];
           end
        end
    end
end