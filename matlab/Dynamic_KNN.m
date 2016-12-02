function [ idx, weight ] = Dynamic_KNN( clusters, sample, LC)
%KNN Summary of this function goes here
%   clusers:    anchor_points: n * p
%   sample:     1 * p
%   k:          number of nearest anchor points to be found
%   idx:        nearest anchor points index 1 * k
%   gamma:      weights 1 * k
% beta = 1.0;

[num_sample, ~] = size(clusters);
D = EuDist2(sample,clusters,0);
D = D*LC;
[D, idx] = sort(D);
lam = D(1)+1;

tmp_dist = 0.0;
tmp_dist_2 = 0.0;

% set threshold to cut off some insignificant anchor points
threshold = 0;

k = 0;

while true
    
%     fprintf('%d epoch Difference: %.4f\n', k, lam - D(k+1));
    if (k>num_sample-1) || (lam - D(k+1) <= threshold)
        break;
    end
    k = k + 1;
    tmp_dist = tmp_dist + D(k);
    tmp_dist_2 = tmp_dist_2 + D(k)^2;
    lam = (tmp_dist + sqrt(k + tmp_dist^2 - k*tmp_dist_2))/k;
end

weight = repmat(lam, 1, k) - D(1:k);
% weight = exp(weight);
idx = idx(1:k);
% weight = weight ./ sum(weight);


% idx = idx(1:k);
% D = D(1:k);
% weight = exp(-beta * D);
end
