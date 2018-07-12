function [model] = constantInputRealPolesEstimation(Y,dynOrder,forcePCS,nullBD,outputUnderRank)

model=sPCAv8(Y,dynOrder,forcePCS,nullBD,outputUnderRank);
