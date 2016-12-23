% load training data
% train_X, train_Y
% load('training_data_100k');
% load('test_data_100k');
[num_sample, ~] = size(train_X);
p = max(train_X(:,2));


y_max = max(train_Y);
y_min = min(train_Y);

% parameters  
iter_num = 1 ;
learning_rate = 0.1;
factors_num = 10;
reg_w = 1e-3;
reg_v = 1e-3;

% locally linear
% anchor points
anchors_num = 50;

% knn
nearest_neighbor = 10;
 
beta = 1.0;

bcon_llfm = zeros(1,iter_num);
sumD_llfm = zeros(1,iter_num);



rmse_llfm_test = zeros(1, iter_num);
rmse_llfm_train = zeros(1,iter_num);

% random pick
% idx = randperm(num_sample);
% % anchors = train_X(idx(1:anchors_num), :);
% anchors = 0.01* rand(anchors_num, p);


for i=1:iter_num
    % do shuffle
    re_idx = randperm(num_sample);
    X_train = train_X(re_idx,:);
    Y_train = train_Y(re_idx);
    
    w0 = rand(1, anchors_num);
    W = rand(p,anchors_num);
    V = rand(p,factors_num,anchors_num);

    mse_llfm_sgd = zeros(1,num_sample);
    loss = zeros(1,num_sample);
    
    

    % get anchor points
    fprintf('Start K-means...\n');
    [~, anchors, bcon_llfm(i), SD, ~] = litekmeans(sparse_matrix(X_train), anchors_num);
    sumD_llfm(i) = sum(SD);
    
%     anchors = 0.01* rand(anchors_num, p);
    fprintf('K-means done..\n');
    
    % SGD
    tic;
    for j=1:num_sample
        if mod(j,1e3)==0
            toc;
            tic;
            fprintf('%d epoch---processing %dth sample\n', i, j);
        end
        
%         X = X_train(j,:);
%         y = Y_train(j,:);

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

            %             tmp = sum(repmat(X',1,factors_num).*temp_V);
            %             y_anchor(k) = (sum(tmp.^2) - sum(sum(repmat(X'.^2,1,factors_num).*(temp_V.^2))))/2 + w0(anchor_idx(k)) + X*W(:,anchor_idx(k));

%             temp_w0 = w0(anchor_idx(k));
%             temp_W = W(:,anchor_idx(k));
            y_anchor(k) = sum(temp_V(1,:).*temp_V(2,:)) + w0(anchor_idx(k)) + sum(W(feature_idx,anchor_idx(k)));
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
        
%         idx = (i-1)*num_sample + j;
%         loss(idx) = err^2;
%         mse_llfm_sgd(idx) = sum(loss)/idx;
        idx = j;
        if idx==1
            mse_llfm_sgd(idx) = err^2;
        else
            mse_llfm_sgd(idx) = (mse_llfm_sgd(idx-1) * (idx - 1) + err^2)/idx;
        end
        
        rmse_llfm_train(i) = mse_llfm_sgd(idx)^0.5;
        
        % update parameters
        tmp_w0 = w0(anchor_idx);
        w0(anchor_idx) = tmp_w0 - learning_rate * gamma .* (2 * err);
%         tmp_W = W(:,anchor_idx);
%         W(:,anchor_idx) = tmp_W - learning_rate * repmat(gamma,p,1) .* (2*err*repmat(X',[1,nearest_neighbor]) + 2*reg_w*tmp_W);
        tmp_W = W(feature_idx,anchor_idx);
        W(feature_idx,anchor_idx) =  tmp_W - learning_rate * (2*repmat(gamma,2,1)*err + 2*reg_w*tmp_W);
        
        for k=1:nearest_neighbor
%             temp_V = squeeze(V(:,:,anchor_idx(k)));
              temp_V = squeeze(V(feature_idx,:,anchor_idx(k)));
%             V(:,:,anchor_idx(k)) = temp_V - learning_rate * gamma(k) * (2*err*(repmat(X',1,factors_num).*(repmat(X*temp_V,p,1)-repmat(X',1,factors_num).*temp_V)) + 2*reg_v*squeeze(temp_V));
              V(feature_idx,:,anchor_idx(k)) = ...
                  temp_V - learning_rate * ...
                  (2*err*gamma(k)*(repmat(sum(temp_V),2,1)- temp_V) + 2*reg_v*temp_V);
        end
        
%         V(:,:,anchor_idx) = temp_V - learning_rate * ...
%             (2 * err * repmat(gamma,p,1,nearest_neighbor) .* X_ .*(repmat(sum(tmp,1),p,1,1)-tmp)  + 2*reg_v*temp_V);
        
%         V = V - learning_rate * (2*err*(repmat(X',1,factors_num).*(repmat(X*temp_V,p,1)-repmat(X',[nearest_neighbor, 1,factors_num]).*V)) + 2*reg_v*V);
    end
    
    % validate

    mse_llfm_test = 0.0;
    [num_sample_test, ~] = size(test_X);
    tic;
    for j=1:num_sample_test
        if mod(j,1e3)==0
            toc;
            tic;
            fprintf('%d epoch(validation)---processing %dth sample\n',i, j);
         end

%         X = test_X(j,:);
%         y = test_Y(j,:);

        X = zeros(1, p);
        feature_idx = test_X(j,:);
        X(feature_idx) = 1;
        y = test_Y(j,:);

        % pick anchor points
        [anchor_idx, weight] = knn(anchors, X, nearest_neighbor, beta);
        gamma = weight/sum(weight);

        y_anchor = zeros(1, nearest_neighbor);
%         for k=1:nearest_neighbor
%             temp_V = squeeze(V(:,:,anchor_idx(k)));
%             tmp = sum(repmat(X',1,factors_num).*temp_V);
%             y_anchor(k) = (sum(tmp.^2) - sum(sum(repmat(X'.^2,1,factors_num).*(temp_V.^2))))/2 + w0(anchor_idx(k)) + X*W(:,anchor_idx(k));
%         end
        for k=1:nearest_neighbor
            temp_V = squeeze(V(feature_idx,:,anchor_idx(k)));
            y_anchor(k) = sum(temp_V(1,:).*temp_V(2,:)) + w0(anchor_idx(k)) + sum(W(feature_idx,anchor_idx(k)));
        end

        y_predict = gamma * y_anchor';
        % prune
%         if y_predict < y_min
%             y_predict = y_min;
%         end
%          
%         if y_predict > y_max
%             y_predict = y_max;
%         end
        
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