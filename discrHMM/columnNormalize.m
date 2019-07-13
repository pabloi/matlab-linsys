function p=columnNormalize(p)
    %Normalization across columns
    p=p./sum(p,1);
end
