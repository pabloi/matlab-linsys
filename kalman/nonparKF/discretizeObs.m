function discreteObs=discretizeObs(observations,Nbins,range)
%Discretizes continuous-valued observations into bins, and returns, for each observation, its corresponding bin.

%TODO: allow for non-uniformly sampled discretization, perhaps percentile-based
if nargin<3 || numel(range)~=2
  range=[min(observations) max(observations)];
end
if nargin<2 || isempty(Nbins)
  Nbins=100;
end
discreteObs=ceil(Nbins*(observations-range(1))/diff(range)); %If observations are uniformly distributed in range, then the discretization is uniformly distributed in [1:Nbins].
if Nbins<=255
  discreteObs=uint8(discreteObs);
elseif Nbins<=65535
  discreteOBs=uint16(discreteObs);
else
  error('Too many observation bins, this will not work. Aborting.')
end

%Avoid overflow by saturating: (if range was given, no guarantee that there are no samples outside range).
discreteObs(discreteObs<1)=1;
discreteObs(discreteObs>Nbins)=Nbins;
end
