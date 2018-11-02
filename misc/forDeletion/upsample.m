function [A,B,Q]=upsample(m,A,B,Q)
%Takes a discrete STATIONARY linear SSM and returns a new system that
%predicts takes m steps for every one step of the original system.
%See also: downsample
A=downsample(1/m,A,B,Q);
end