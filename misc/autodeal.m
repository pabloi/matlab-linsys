function structure = autodeal(varargin)

for i=1:length(varargin)
    fieldName=inputname(i);
   structure.(fieldName)=varargin{i}; 
end
end