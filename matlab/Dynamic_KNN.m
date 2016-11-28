function [ idx, weight ] = Dynamic_KNN( clusters, sample)
%KNN Summary of this function goes here
%   clusers:    anchor_points: n * p
%   sample:     1 * p
%   k:          number of nearest anchor points to be found
%   idx:        nearest anchor points index 1 * k
%   gamma:      weights 1 * k
% beta = 1.0;

noise_ratio = 2;
[num_sample, ~] = size(clusters);
D = EuDist2(sample,clusters);
D = D*noise_ratio;
[D, idx] = sort(D);
lam = D(1)+1;

tmp_dist = 0.0;
tmp_dist_2 = 0.0;

for k=1:num_sample-1
    
%     fprintf('%d epoch Difference: %.4f\n', k, lam - D(k+1));
    if lam <= D(k+1)
        break;
    end
    tmp_dist = tmp_dist + D(k);
    tmp_dist_2 = tmp_dist_2 + D(k)^2;
    lam = (tmp_dist + sqrt(k + tmp_dist^2 - k*tmp_dist_2))/k;
end

weight = repmat(lam, 1, k) - D(1:k);
idx = idx(1:k);
% weight = weight ./ sum(weight);


% idx = idx(1:k);
% D = D(1:k);
% weight = exp(-beta * D);
end
