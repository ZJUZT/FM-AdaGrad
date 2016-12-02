% load training data
% train_X, train_Y
load('training_data');
load('test_data');
[num_sample, p] = size(train_X);
y_max = max(train_Y);
y_min = min(train_Y);

% parameters 
iter_num = 1;
learning_rate = 0.1;
factors_num = 10;
reg_w = 0.001;
reg_v = 0.001;

% locally linear
% anchor points
anchors_num = 100;

% knn
nearest_neighbor = 10;

w0 = rand(1, anchors_num);
W = rand(p,anchors_num);
V = rand(p,factors_num,anchors_num);

mse_llfm_sgd = zeros(1,iter_num*num_sample);
loss = zeros(1,iter_num*num_sample);

rmse_llfm_test = zeros(1, iter_num);

% get anchor points
fprintf('Start K-means...\n');
% [~, anchors, ~, ~, ~] = litekmeans(train_X, anchors_num);

% random pick
% idx = randperm(num_sample);
% anchors = train_X(idx(1:anchors_num), :);
anchors = 0.01* rand(anchors_num, p);
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

%         temp_V = V(:,:,anchor_idx);
%         X_ = repmat(X', [1, factors_num, nearest_neighbor]);
%         tmp = X_.*temp_V;
%         factor_part = gamma * (squeeze(sum(sum(tmp.^2,1),2) -(sum(sum((X_.^2).*(temp_V.*2),1),2))));
        
        
        
        y_predict = gamma * y_anchor';
        
%         fprintf('%f\n', y_predict);
        
        % prune
%         if y_predict < y_min
%             y_predict = y_min;
%         end
%         
%         if y_predict > y_max
%             y_predict = y_max;
%         end
        
        err = y_predict - y;
        
        idx = (i-1)*num_sample + j;
        loss(idx) = err^2;
        mse_llfm_sgd(idx) = sum(loss)/idx;
        
        % update parameters
        tmp_w0 = w0(anchor_idx);
        w0(anchor_idx) = tmp_w0 - learning_rate * gamma .* (2 * err + 2*reg_w*tmp_w0);
        tmp_W = W(:,anchor_idx);
        W(:,anchor_idx) = tmp_W - learning_rate * repmat(gamma,p,1) .* (2*err*repmat(X',[1,nearest_neighbor]) + 2*reg_w*tmp_W);
        
        for k=1:nearest_neighbor
            temp_V = squeeze(V(:,:,anchor_idx(k)));
            V(:,:,anchor_idx(k)) = temp_V - learning_rate * gamma(k) * (2*err*(repmat(X',1,factors_num).*(repmat(X*temp_V,p,1)-repmat(X',1,factors_num).*temp_V)) + 2*reg_v*squeeze(temp_V));
        end
        
%         V(:,:,anchor_idx) = temp_V - learning_rate * ...
%             (2 * err * repmat(gamma,p,1,nearest_neighbor) .* X_ .*(repmat(sum(tmp,1),p,1,1)-tmp)  + 2*reg_v*temp_V);
        
%         V = V - learning_rate * (2*err*(repmat(X',1,factors_num).*(repmat(X*temp_V,p,1)-repmat(X',[nearest_neighbor, 1,factors_num]).*V)) + 2*reg_v*V);
    end
    
    % validate

    mse_llfm_test = 0.0;
    [num_sample_test, p] = size(test_X);
    tic;
    for j=1:num_sample_test
        if mod(j,1000)==0
            toc;
            tic;
            fprintf('processing %dth sample\n', j);
         end

        X = test_X(j,:);
        y = test_Y(j,:);

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
        mse_llfm_test = mse_llfm_test + err.^2;
    end

    rmse_llfm_test(i) = (mse_llfm_test / num_sample_test).^0.5;
end

%%


%%
% plot
plot(mse_llfm_sgd.^0.5);
xlabel('Number of samples seen');
ylabel('RMSE');
grid on;