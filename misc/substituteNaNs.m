function y=substituteNaNs(y)
%Acts column-wise, replacing NaNs with a linearly interpolated value
for i=1:size(y,2)
  notNaN=~isnan(y(:,i));
  if any(~notNaN)
    y(~notNaN,i)=interp1(find(notNaN),y(notNaN,i),find(~notNaN));
  end
end
