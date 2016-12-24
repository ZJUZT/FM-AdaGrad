% load training data
% train_X, train_Y
% load('training_data_1m');
% load('test_data_1m'); 
[num_sample, ~] = size(train_X);
p = max(train_X(:,2));

y_max = max(train_Y);
y_min = min(train_Y);

% parameters 
iter_num = 1 ;
learning_rate = 0.1;
learning_rate_anchor = 1e-2; 
factors_num = 10;
reg_w = 1e-3; 
reg_v = 1e-3;

% locally linear
% anchor points
anchors_num = 30;

beta = 1;

bcon_dallfm = zeros(1,iter_num);
sumD_dallfm = zeros(1,iter_num);

% knn
nearest_neighbor = 20;

rmse_dallfm_test = zeros(1, iter_num);

rmse_dallfm_train = zeros(1, iter_num);

for i=1:iter_num
    % do shuffle
    re_idx = randperm(num_sample);
    X_train = train_X(re_idx,:);
    Y_train = train_Y(re_idx);
    
    mse_da_llfm_sgd = zeros(1,num_sample);
    loss = zeros(1,num_sample);
    w0 = rand(1, anchors_num);
    W = rand(p,anchors_num);
    V = rand(p,factors_num,anchors_num);
    
    % get anchor points
    fprintf('Start K-means...\n');
%     [~, anchors, bcon_dallfm(i), SD, ~] = litekmeans(sparse_matrix(X_train), anchors_num);
%     sumD_dallfm(i) = sum(SD);
    anchors = 0.01* rand(anchors_num, p);
    fprintf('K-means done..\n');
    
    % SGD
    tic;
    for j=1:num_sample
        if mod(j,1e3)==0
            toc;
            tic;
            fprintf('%d epoch---processing %dth sample\n', i, j);
        end

        X = zeros(1, p);
        feature_idx = X_train(j,:);
        X(feature_idx) = 1;
        y = Y_train(j,:);
        
        % pick anchor points
        [anchor_idx, weight] = knn(anchors, X, nearest_neighbor, beta);
        gamma = weight/sum(weight);
        
        y_anchor = zeros(1, nearest_neighbor);
        for k=1:nearest_neighbor
            temp_V = squeeze(V(feature_idx,:,anchor_idx(k)));
            y_anchor(k) = sum(temp_V(1,:).*temp_V(2,:)) + w0(anchor_idx(k)) + sum(W(feature_idx,anchor_idx(k)));
        end

        y_predict = gamma * y_anchor';
        err = y_predict - y;
        
        idx=j;
        if idx==1
            mse_da_llfm_sgd(idx) = err^2;
        else
            mse_da_llfm_sgd(idx) = (mse_da_llfm_sgd(idx-1) * (idx - 1) + err^2)/idx;
        end
        
        rmse_dallfm_train(i) = mse_da_llfm_sgd(idx)^0.5;
        
        % update parameters
        tmp_w0 = w0(anchor_idx);
        w0(anchor_idx) = tmp_w0 - learning_rate * gamma .* (2 * err);
        
        tmp_W = W(feature_idx,anchor_idx);
        W(feature_idx,anchor_idx) =  tmp_W - learning_rate * (2*err*repmat(gamma,2,1) + 2*reg_w*tmp_W);

        for k=1:nearest_neighbor
            temp_V = squeeze(V(feature_idx,:,anchor_idx(k)));
            
            V(feature_idx,:,anchor_idx(k)) = ...
                  temp_V - learning_rate *...
                  (2*err* gamma(k) * ((repmat(sum(temp_V),2,1))- temp_V) + 2*reg_v*temp_V);
        end
        
        % update anchor points
        s = 2 * beta * (repmat(X, nearest_neighbor, 1) - anchors(anchor_idx, :)).*repmat(weight, p, 1)';
        base = -s * sum(weight.*y_anchor);
        base = base + repmat(y_anchor',1,p).* s*sum(weight);
        anchors(anchor_idx,:) = anchors(anchor_idx,:) - learning_rate_anchor * 2 * err * base/(sum(weight).^2);

    end
    
    
    % validate
    mse_dallfm_test = 0.0;
    [num_sample_test, ~] = size(test_X);
    tic;
    for k=1:num_sample_test
        if mod(k,1000)==0
            toc;
            tic;
            fprintf('%d epoch(validation)---processing %dth sample\n',i, k);
         end

        X = zeros(1, p);
        feature_idx = test_X(k,:);
        X(feature_idx) = 1;
        y = test_Y(k,:);

        % pick anchor points
        [anchor_idx, weight] = knn(anchors, X, nearest_neighbor, beta);
        gamma = weight/sum(weight);

        y_anchor = zeros(1, nearest_neighbor);

        for n=1:nearest_neighbor
            temp_V = squeeze(V(feature_idx,:,anchor_idx(n)));
            y_anchor(n) = sum(temp_V(1,:).*temp_V(2,:)) + w0(anchor_idx(n)) + sum(W(feature_idx,anchor_idx(n)));
        end

        y_predict = gamma * y_anchor';

        err = y_predict - y;
        mse_dallfm_test = mse_dallfm_test + err.^2;
    end

    rmse_dallfm_test(i) = (mse_dallfm_test / num_sample_test)^0.5 ;
end

%%
% validate

%%
% plot
plot(mse_da_llfm_sgd.^0.5);
xlabel('Number of samples seen');
ylabel('RMSE');
grid on;