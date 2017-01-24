% load training data
% train_X, train_Y
% load('training_data_100k');
% load('test_data_100k');
[num_sample, p] = size(train_X);
% p = max(train_X(:,2));


y_max = max(train_Y);
y_min = min(train_Y);

% parameters  
iter_num = 1;
learning_rate = 1e-2;
factors_num = 10;
reg_w = 0;
reg_v = 0; 

% locally linear
% anchor points
anchors_num = 10;


epoch = 1;

% knn
nearest_neighbor = 5;
beta = 0.1;

bcon_llfm = zeros(1,iter_num);
sumD_llfm = zeros(1,iter_num);



rmse_llfm_test = zeros(iter_num, epoch);
rmse_llfm_train = zeros(iter_num, epoch);
accuracy_llfm = zeros(iter_num, epoch);

% random pick
% idx = randperm(num_sample);
% % anchors = train_X(idx(1:anchors_num), :);
% anchors = 0.01* rand(anchors_num, p);


for i=1:iter_num
    % do shuffle
    
    
    w0 = 0.1*randn(1, anchors_num);
    W = 0.1*randn(p,anchors_num);
    V = 0.1*randn(p,factors_num,anchors_num);

    mse_llfm_sgd = zeros(1,num_sample);
    loss = zeros(1,epoch*num_sample);
    
    

    % get anchor points
    fprintf('Start K-means...\n');
%     [~, anchors, bcon_llfm(i), SD, ~] = litekmeans(sparse_matrix(train_X), anchors_num);
    [~, anchors, bcon_llfm(i), SD, ~] = litekmeans(train_X, anchors_num);
    sumD_llfm(i) = sum(SD);
    
%     anchors = 0.01* rand(anchors_num, p);
    fprintf('K-means done..\n');
    
    % SGD
    tic;
    
    for t=1:epoch
        
        re_idx = randperm(num_sample);
        X_train = train_X(re_idx,:);
        Y_train = train_Y(re_idx);
        
        for j=1:num_sample
            if mod(j,1e3)==0
                toc;
                tic;
                fprintf('%d iter(%d epoch)---processing %dth sample\n', i,t,j);
            end

            X = X_train(j,:);
            y = Y_train(j,:);

%             X = zeros(1, p);
%             feature_idx = X_train(j,:);
%             X(feature_idx) = 1;
% 
%             y = Y_train(j,:);

            % pick anchor points
            [anchor_idx, weight] = knn(anchors, X, nearest_neighbor, beta);         
            gamma = weight/sum(weight);

            y_anchor = zeros(1, nearest_neighbor);
            for k=1:nearest_neighbor
%                 temp_V = squeeze(V(feature_idx,:,anchor_idx(k)));
                
                temp_V = V(:,:,anchor_idx(k));
                tmp = sum(repmat(X',1,factors_num).*temp_V);
                y_anchor(k) = (sum(tmp.^2) - sum(sum(repmat(X'.^2,1,factors_num).*(temp_V.^2))))/2 + w0(anchor_idx(k)) + X*W(:,anchor_idx(k));

    %             temp_w0 = w0(anchor_idx(k));
    %             temp_W = W(:,anchor_idx(k));
%                 y_anchor(k) = sum(temp_V(1,:).*temp_V(2,:)) + w0(anchor_idx(k)) + sum(W(feature_idx,anchor_idx(k)));
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

            % err = y_predict - y;
            err_c = sigmf(y*y_predict,[1,0]);

    %         idx = (i-1)*num_sample + j;
    %         loss(idx) = err^2;
    %         mse_llfm_sgd(idx) = sum(loss)/idx;
%             idx = (t-1)*num_sample + j;
            idx = j;
            if idx==1
                mse_llfm_sgd(idx) = -log(err_c);
                % mse_llfm_sgd(idx) = err^2;
            else
                mse_llfm_sgd(idx) = (mse_llfm_sgd(idx-1) * (idx - 1) -log(err_c))/idx;
                % mse_llfm_sgd(idx) = (mse_llfm_sgd(idx-1) * (idx - 1) + err^2)/idx;
            end

            % rmse_llfm_train(i, t) = mse_llfm_sgd(idx)^0.5;
            rmse_llfm_train(i, t) = mse_llfm_sgd(idx);

            % update parameters
            tmp_w0 = w0(anchor_idx);
            w0(anchor_idx) = tmp_w0 - learning_rate * gamma .* (err_c-1)*y;
            tmp_W = W(:,anchor_idx);
            W(:,anchor_idx) = tmp_W - learning_rate * repmat(gamma,p,1) .* ((err_c-1)*y*repmat(X',[1,nearest_neighbor]) + 2*reg_w*tmp_W);
%             tmp_W = W(feature_idx,anchor_idx);
%             W(feature_idx,anchor_idx) =  tmp_W - learning_rate * repmat(gamma,2,1).*(2*err + 2*reg_w*tmp_W);

            for k=1:nearest_neighbor
                temp_V = squeeze(V(:,:,anchor_idx(k)));
%                   temp_V = squeeze(V(feature_idx,:,anchor_idx(k)));
                V(:,:,anchor_idx(k)) = temp_V - learning_rate * gamma(k) * ((err_c-1)*y*(repmat(X',1,factors_num).*(repmat(X*temp_V,p,1)-repmat(X',1,factors_num).*temp_V)) + 2*reg_v*temp_V);
%                   V(feature_idx,:,anchor_idx(k)) = ...
%                       temp_V - learning_rate * gamma(k)* ...
%                       (2*err*(repmat(sum(temp_V),2,1)- temp_V) + 2*reg_v*temp_V);
            end

    %         V(:,:,anchor_idx) = temp_V - learning_rate * ...
    %             (2 * err * repmat(gamma,p,1,nearest_neighbor) .* X_ .*(repmat(sum(tmp,1),p,1,1)-tmp)  + 2*reg_v*temp_V);

    %         V = V - learning_rate * (2*err*(repmat(X',1,factors_num).*(repmat(X*temp_V,p,1)-repmat(X',[nearest_neighbor, 1,factors_num]).*V)) + 2*reg_v*V);
        end
    
    
        % validate

        mse_llfm_test = 0.0;
        correct_num = 0;
        [num_sample_test, ~] = size(test_X);
        tic;
        for j=1:num_sample_test
            if mod(j,1e3)==0
                toc;
                tic;
                fprintf('%d epoch(validation)---processing %dth sample\n',i, j);
             end

            X = test_X(j,:);
            y = test_Y(j,:);

%             X = zeros(1, p);
%             feature_idx = test_X(j,:);
%             X(feature_idx) = 1;
%             y = test_Y(j,:);

            % pick anchor points
            [anchor_idx, weight] = knn(anchors, X, nearest_neighbor, beta);
            gamma = weight/sum(weight);

            y_anchor = zeros(1, nearest_neighbor);
            for k=1:nearest_neighbor
                temp_V = squeeze(V(:,:,anchor_idx(k)));
                tmp = sum(repmat(X',1,factors_num).*temp_V);
                y_anchor(k) = (sum(tmp.^2) - sum(sum(repmat(X'.^2,1,factors_num).*(temp_V.^2))))/2 + w0(anchor_idx(k)) + X*W(:,anchor_idx(k));
            end
%             for k=1:nearest_neighbor
%                 temp_V = squeeze(V(feature_idx,:,anchor_idx(k)));
%                 y_anchor(k) = sum(temp_V(1,:).*temp_V(2,:)) + w0(anchor_idx(k)) + sum(W(feature_idx,anchor_idx(k)));
%             end

            y_predict = gamma * y_anchor';
            % prune
    %         if y_predict < y_min
    %             y_predict = y_min;
    %         end
    %          
    %         if y_predict > y_max
    %             y_predict = y_max;
    %         end
            if (y_predict>=0 && y==1) || (y_predict<0&&y==-1)
                correct_num = correct_num + 1;
            end

            % err = y_predict - y;
            err_c = sigmf(y*y_predict,[1,0]);
            % mse_llfm_test = mse_llfm_test + err.^2;
            mse_llfm_test = mse_llfm_test - log(err_c);
        end

        accuracy_llfm(i,t) = correct_num/num_sample_test;
        rmse_llfm_test(i, t) = (mse_llfm_test / num_sample_test);
    end
end

%%


%%
% plot
plot(mse_llfm_sgd,'DisplayName','LLFM\_Train');
legend('-DynamicLegend');
xlabel('Number of samples seen');
ylabel('RMSE'); 
grid on;
hold on;

%%
plot(rmse_llfm_train,'DisplayName','LLFM\_Train');
legend('-DynamicLegend');
hold on;
plot(rmse_llfm_test,'DisplayName','LLFM\_Test');
legend('-DynamicLegend');
xlabel('epoch');
ylabel('RMSE');
% legend('LLFM\_Train','LLFM\_Test');
title('LLFM\_SGD');
grid on;