% load training data
% train_X, train_Y
load('training_data');
[num_sample, p] = size(train_X);
y_max = max(train_Y);
y_min = min(train_Y);

% parameters
iter_num = 1;
learning_rate = 0.01;
factors_num = 10;
reg_w = 0.1;
reg_v = 0.01;

% locally linear
% anchor points
anchors_num = 100;

% knn
nearest_neighbor = 10;

w0 = rand(1, anchors_num);
W = rand(p,anchors_num);
V = rand(p,factors_num,anchors_num);

mse_da_llfm_sgd = zeros(1,iter_num*num_sample);
loss = zeros(1,iter_num*num_sample);

% get anchor points
fprintf('Start K-means...\n');

% initial anchor points
[~, anchors, ~, ~, ~] = litekmeans(train_X, anchors_num);
fprintf('K-means done..\n');

for i=1:iter_num
    % do shuffle
    re_idx = randperm(num_sample);
    X_train = train_X(re_idx,:);
    Y_train = train_Y(re_idx);
    
    % SGD
    tic;
    for j=1:num_sample
        if mod(j,1000)==0
            toc;
            tic;
            fprintf('processing %dth sample\n', j);
        end
        
        X = X_train(j,:);
        y = Y_train(j,:);
        
        % pick anchor points
        [anchor_idx, weight] = knn(anchors, X, nearest_neighbor);
        gamma = weight/sum(weight);
        
        y_anchor = zeros(1, nearest_neighbor);
        for k=1:nearest_neighbor
            temp_V = squeeze(V(:,:,anchor_idx(k)));
            tmp = sum(repmat(X',1,factors_num).*temp_V);
            y_anchor(k) = (sum(tmp.^2) - sum(sum(repmat(X'.^2,1,factors_num).*(temp_V.^2))))/2 + w0(anchor_idx(k)) + X*W(:,anchor_idx(k));
        end

        
        y_predict = gamma * y_anchor';
        
%         fprintf('%f\n', y_predict);
        
        % prune
        if y_predict < y_min
            y_predict = y_min;
        end
        
        if y_predict > y_max
            y_predict = y_max;
        end
        
        err = y_predict - y;
        
        idx = (i-1)*num_sample + j;
        loss(idx) = err^2;
        mse_da_llfm_sgd(idx) = sum(loss)/idx;
        
        % update parameters
        tmp_w0 = w0(anchor_idx);
        w0(anchor_idx) = tmp_w0 - learning_rate * gamma .* (2 * err + 2*reg_w*tmp_w0);
        tmp_W = W(:,anchor_idx);
        W(:,anchor_idx) = tmp_W - learning_rate * repmat(gamma,p,1) .* (2*err*repmat(X',[1,nearest_neighbor]) + 2*reg_w*tmp_W);
        
        for k=1:nearest_neighbor
            temp_V = squeeze(V(:,:,anchor_idx(k)));
            V(:,:,anchor_idx(k)) = temp_V - learning_rate * gamma(k) * ...
                (2*err*(repmat(X',1,factors_num).*(repmat(X*temp_V,p,1)-repmat(X',1,factors_num).*temp_V)) + 2*reg_v*squeeze(temp_V));
            tmp = anchors(anchor_idx(k), :);
            
            
            s = 2 * (repmat(X, nearest_neighbor, 1) - anchors(anchor_idx, :)).*repmat(weight, p, 1)';
            delt = -s .* repmat(weight, p, 1)';
            delt(k, :) = delt(k,:) + s(k,:) * sum(weight);
            delt = delt / (sum(weight).^2);
            
            anchors(anchor_idx(k), :) = tmp - learning_rate * 2 * err * y_anchor*delt;
        end
        
        % update anchor points
        
        
%         for k = 1:nearest_neighbor
%             tmp = anchors(anchor_idx(k), :);
%             s = 2 * (repmat(X, nearest_neighbor, 1) - anchors(anchor_idx, :)).*repmat(weight, p, 1)';
%             delt = -s .* repmat(weight, p, 1)';
%             delt(k, :) = delt(k,:) + s(k,:) * sum(weight);
%             delt = delt / (sum(weight).^2);
%             
%             anchors(anchor_idx(k), :) = tmp - learning_rate * 2 * err * y_anchor*delt;
%         end
        
    end
end

%%
% validate

mse_dallfm_test = 0.0;
[num_sample_test, p] = size(test_X);
tic;
for i=1:num_sample_test
    if mod(i,1000)==0
        toc;
        tic;
        fprintf('processing %dth sample\n', i);
     end

    X = test_X(i,:);
    y = test_Y(i,:);

    % pick anchor points
    [anchor_idx, weight] = knn(anchors, X, nearest_neighbor);
    gamma = weight/sum(weight);

    y_anchor = zeros(1, nearest_neighbor);
    for k=1:nearest_neighbor
        temp_V = squeeze(V(:,:,anchor_idx(k)));
        tmp = sum(repmat(X',1,factors_num).*temp_V);
        y_anchor(k) = (sum(tmp.^2) - sum(sum(repmat(X'.^2,1,factors_num).*(temp_V.^2))))/2 + w0(anchor_idx(k)) + X*W(:,anchor_idx(k));
    end
    
    y_predict = gamma * y_anchor';
    err = y_predict - y;
    mse_dallfm_test = mse_dallfm_test + err.^2;
end

mse_dallfm_test = mse_dallfm_test / num_sample_test;

%%
% plot
plot(mse_da_llfm_sgd);
xlabel('Number of samples seen');
ylabel('MSE');
grid on;