% load training data
% train_X, train_Y

% fully adaptive (KNN && anchor points)

load('training_data_100k');
load('test_data_100k'); 
[num_sample, ~] = size(train_X);
p = max(train_X(:,2));

y_max = max(train_Y);
y_min = min(train_Y);

% parameters 
iter_num = 1;
learning_rate = 0.1;
learning_rate_anchor = 0.001;
factors_num = 10;
reg_w = 0.001;
reg_v = 0.001;

% locally linear
% anchor points
anchors_num = 100;

% Lipschitz to noise ratio
% control the number of neighbours
LC = 5;

% knn
% nearest_neighbor = 10;
num_nn = 0;
num_nn_batch = 0;

w0 = rand(1, anchors_num);
W = rand(p,anchors_num);
V = rand(p,factors_num,anchors_num);

mse_dadk_llfm_sgd = zeros(1,iter_num*num_sample);
loss = zeros(1,iter_num*num_sample);

rmse_dadk_llfm_test = zeros(1, iter_num);

% get anchor points
fprintf('Start K-means...\n');

% initial anchor points
% [~, anchors, ~, ~, ~] = litekmeans(train_X, anchors_num);
% idx = randperm(num_sample);
% anchors = train_X(idx(1:anchors_num), :);
anchors = 0.01*rand(anchors_num, p);
fprintf('K-means done..\n');

for i=1:iter_num
    % do shuffle
    re_idx = randperm(num_sample);
    X_train = train_X(re_idx,:);
    Y_train = train_Y(re_idx);
    
    % SGD
    tic;
    for j=1:num_sample
        if mod(j,100)==0
            toc;
            tic;
            
            fprintf('%d epoch---processing %dth sample\n', i, j);
            fprintf('batch average value of K in KNN is %.2f\n', num_nn_batch/100);
            fprintf('overall average value of K in KNN is %.2f\n', num_nn/((i-1)*num_sample + j));
            
            num_nn_batch = 0;
            
        end
        
%         X = X_train(j,:);
%         y = Y_train(j,:);

        X = zeros(1, p);
        feature_idx = X_train(j,:);
        X(feature_idx) = 1;
        y = Y_train(j,:);
        
        % pick anchor points
%         [anchor_idx, weight] = knn(anchors, X, nearest_neighbor);

        [anchor_idx, weight] = Dynamic_KNN(anchors, X, LC);
        gamma = weight/sum(weight);
        nearest_neighbor = length(anchor_idx);
        
        num_nn = num_nn + nearest_neighbor;
        num_nn_batch = num_nn_batch + nearest_neighbor;
        
        y_anchor = zeros(1, nearest_neighbor);
        for k=1:nearest_neighbor
%             temp_V = squeeze(V(:,:,anchor_idx(k)));
%             tmp = sum(repmat(X',1,factors_num).*temp_V);
%             y_anchor(k) = (sum(tmp.^2) - sum(sum(repmat(X'.^2,1,factors_num).*(temp_V.^2))))/2 + w0(anchor_idx(k)) + X*W(:,anchor_idx(k));
            temp_V = squeeze(V(feature_idx,:,anchor_idx(k)));
            y_anchor(k) = sum(temp_V(1,:).*temp_V(2,:)) + w0(anchor_idx(k)) + sum(W(feature_idx,anchor_idx(k)));
        end

        
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
%         loss(idx) = err^2;
%         mse_dadk_llfm_sgd(idx) = sum(loss)/idx;
        if idx==1
            mse_dadk_llfm_sgd(idx) = err^2;
        else
            mse_dadk_llfm_sgd(idx) = (mse_dadk_llfm_sgd(idx-1) * (idx - 1) + err^2)/idx;
        end
        
        % update parameters
        tmp_w0 = w0(anchor_idx);
        w0(anchor_idx) = tmp_w0 - learning_rate * gamma .* (2 * err + 2*reg_w*tmp_w0);
%         tmp_W = W(:,anchor_idx);
%         W(:,anchor_idx) = tmp_W - learning_rate * repmat(gamma,p,1) .* (2*err*repmat(X',[1,nearest_neighbor]) + 2*reg_w*tmp_W);

        tmp_W = W(feature_idx,anchor_idx);
        W(feature_idx,anchor_idx) =  tmp_W - learning_rate * repmat(gamma,2,1) .* (2*err + 2*reg_w*tmp_W);
        
        
        s = 2 * LC * (repmat(X, nearest_neighbor, 1) - anchors(anchor_idx, :));
        base = -s .* repmat(weight, p, 1)';
        for k=1:nearest_neighbor
%             temp_V = squeeze(V(:,:,anchor_idx(k)));
%             V(:,:,anchor_idx(k)) = temp_V - learning_rate * gamma(k) * ...
%                 (2*err*(repmat(X',1,factors_num).*(repmat(X*temp_V,p,1)-repmat(X',1,factors_num).*temp_V)) + 2*reg_v*squeeze(temp_V));
            temp_V = squeeze(V(feature_idx,:,anchor_idx(k)));
            
            V(feature_idx,:,anchor_idx(k)) = ...
                  temp_V - learning_rate * gamma(k) * ...
                  (2*err*((repmat(sum(temp_V),2,1))- temp_V) + 2*reg_v*temp_V);
            
            % update anchor points
            tmp = anchors(anchor_idx(k), :);
            delt = base;
            delt(k, :) = delt(k,:) + s(k,:) * sum(weight);
            delt = delt / (sum(weight).^2);
            
            anchors(anchor_idx(k), :) = tmp - learning_rate_anchor * 2 * err * y_anchor*delt;
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
    
    
    % validate
    mse_dallfm_test = 0.0;
    [num_sample_test, p] = size(test_X);
    tic;
    for k=1:num_sample_test
        if mod(k,1000)==0
            toc;
            tic;
            fprintf('%d epoch(validation)---processing %dth sample\n',i, k);
         end

%         X = test_X(k,:);
%         y = test_Y(k,:);

        X = zeros(1, p);
        feature_idx = test_X(k,:);
        X(feature_idx) = 1;
        y = test_Y(k,:);

        % pick anchor points
%         [anchor_idx, weight] = knn(anchors, X, nearest_neighbor);
        
        [anchor_idx, weight] = Dynamic_KNN(anchors, X, LC);
        gamma = weight/sum(weight);
        nearest_neighbor = length(anchor_idx);

        y_anchor = zeros(1, nearest_neighbor);
%         for n=1:nearest_neighbor
%             temp_V = squeeze(V(:,:,anchor_idx(n)));
%             tmp = sum(repmat(X',1,factors_num).*temp_V);
%             y_anchor(n) = (sum(tmp.^2) - sum(sum(repmat(X'.^2,1,factors_num).*(temp_V.^2))))/2 + w0(anchor_idx(n)) + X*W(:,anchor_idx(n));
%         end

        for n=1:nearest_neighbor
            temp_V = squeeze(V(feature_idx,:,anchor_idx(n)));
            y_anchor(n) = sum(temp_V(1,:).*temp_V(2,:)) + w0(anchor_idx(n)) + sum(W(feature_idx,anchor_idx(n)));
        end

        y_predict = gamma * y_anchor';
        err = y_predict - y;
        mse_dallfm_test = mse_dallfm_test + err.^2;
    end

    rmse_dadk_llfm_test(i) = (mse_dallfm_test / num_sample_test)^0.5;
end

%%
% validate

%%
% plot
plot(mse_dadk_llfm_sgd.^0.5);
xlabel('Number of samples seen');
ylabel('RMSE');
grid on;