function [ idx, weight ] = knn( clusters, sample, k ,beta)
%KNN Summary of this function goes here
%   clusers:    anchor_points: n * p
%   sample:     1 * p
%   k:          number of nearest anchor points to be found
%   idx:        nearest anchor points index 1 * k
%   gamma:      weights 1 * k


D = EuDist2(sparse(sample),clusters,0);
[D, idx] = sort(D);
idx = idx(1:k);
D = D(1:k);
weight = exp(-beta * D);
end

