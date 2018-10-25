function [ind1,ind2]=bestPairedMatch(vec1,vec2)
  %Returns ind1 and ind2 such that we have a heuristic solution to
  %minimizing norm(vec1(ind1)-vec2(ind2))
  %If vec1 is shorter, then ind1 = 1:length(vec1), otherwise ind2=1:length(vec2)
  %ind1 and ind2 are the same size as vec1 and vec2 respectively

%invertFlag=false;
%  if length(vec1)<length(vec2) %Swapping vectors to have vec2 be shorter
%    invertFlag=true;
%    vecT=vec2;
%    vec2=vec1;
%    vec1=vecT;
%  end

notMatched=true(size(vec1));
ind1=zeros(size(vec1));
i=0;
while i<length(vec2) & any(notMatched)% i=1:length(vec2)
  i=i+1;
  dif=abs(vec1-vec2(i));
  closestInd=find(min(dif(notMatched))==dif & notMatched,1,'first');
  ind1(i)=closestInd;
  notMatched(closestInd)=false;
end
ind1(ind1==0)=find(notMatched);
ind2=1:length(vec2);

%if invertFlag
%  indT=ind2;
%  ind2=ind1;
%  ind1=indT;
%  vecT=vec2;
%  vec2=vec1;
%  vec1=vecT;
%end
