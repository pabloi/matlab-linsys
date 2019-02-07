classdef trainInfo
    %trainInfo is an auxiliary class meant to contain all metadata relating
    %to the training of linsys objects. All fields are optional.
    
    properties
        setHash='';
        method='';
        options=struct();
    end
    
    methods
        function this = trainInfo(hash,methodName,opts)
            if nargin>0
            this.setHash=hash;
            if nargin>1
            this.method=methodName;
            if nargin>2
            this.options=opts;
            end; end; end
        end
    end
end

