function [idx, ctrs, iter_ctrs] = kmeans(X, K)
%KMEANS K-Means clustering algorithm
%
%   Input: X - data point features, n-by-p maxtirx.
%          K - the number of clusters
%
%   OUTPUT: idx  - cluster label
%           ctrs - cluster centers, K-by-p matrix.
%           iter_ctrs - cluster centers of each iteration, K-by-p-by-iter
%                       3D matrix.

% YOUR CODE HERE
[n,p] = size(X);
%iter_ctrs = zeros(K,p,iter);

% may generate identical centroids
% ctrs_idx = randi([1,n],1,K);
ctrs_idx = randperm(n,K);
iter_ctrs(:,:,1) = X(ctrs_idx,:);   %initial centroids
idx = zeros(1,n);   %cluster label



e = 1e-5;
iter = 1;
while true
    dist = EuDist2(X,iter_ctrs(:,:,iter));
    [~,idx] = min(dist,[],2);
    ctrs = zeros(K,p);
    for i=1:K
        ctrs(i,:) = mean(X(idx==i,:));   %update centroids
    end
    if(abs(ctrs - iter_ctrs(:,:,iter))<e)
        break;
    end
    iter = iter+1;
    iter_ctrs(:,:,iter) = ctrs;
end

idx = idx';
end
