  % load training data
% train_X, train_Y
% load('training_data_1m');
% load('test_data_1m'); 
rng('default');
recommendation = 0;
regression = 1;
classification = 2;

% rand('state',1); 
% randn('state',1);

task = classification;      

if task == recommendation
    [num_sample, ~] = size(train_X);
    p = max(train_X(:,2));
else
    [num_sample, p] = size(train_X);
end

% y_max = max(train_Y);
% y_min = min(train_Y);

% parameters 
iter_num = 1;
epoch = 20;

% learning_rate = 5e-2;
% learning_rate_anchor = 5e-2;
factors_num = 10;

% ijcnn
% learning_rate = 5e4;
% t0 = 1e5;
% skip = 1e3;

% learning_rate = 5e-2;
% reg = 1e-3;

% banana
learning_rate = 1e4;
t0 = 1e5;
skip = 1e1;

% ml 100k
% learning_rate = 1e4;
% t0 = 1e5;
% skip = 1e3;

count = skip;

% T = 1e5  ;
% locally linear
% anchor points
anchors_num = 10;

beta = 1  ;

bcon_dallfm = zeros(iter_num, epoch);
sumD_dallfm = zeros(iter_num, epoch);
accuracy_dallfm = zeros(iter_num, epoch);

% knn
nearest_neighbor = 2 ;

rmse_dallfm_test = zeros(iter_num,epoch);

rmse_dallfm_train = zeros(iter_num,epoch);
 
for i=1:iter_num
    
    w0 = zeros(1, anchors_num);
    W = zeros(p,anchors_num);
    V = 0.1*randn(p,factors_num,anchors_num);

    mse_da_llfm_sgd = zeros(1,num_sample);
    loss = zeros(1,num_sample);
    
    % get anchor points
    fprintf('Start K-means...\n');
%     [~, anchors, bcon_dallfm(i), SD, ~] = litekmeans(sparse_matrix(train_X), anchors_num, 'Replicates', 10);
    [~, anchors, ~, SD, ~] = litekmeans(train_X, anchors_num, 'Replicates', 10);
%     sumD_dallfm(i) = sum(SD);
%     anchors = 0.1*randn(anchors_num, p);
    fprintf('K-means done..\n');
    
    % SGD
    tic;
    % do shuffle
    re_idx = randperm(num_sample);
    X_train = train_X(re_idx,:);
    Y_train = train_Y(re_idx);
    
    for t=1:epoch

    
%         X_train = train_X;
%         Y_train = train_Y;

%         re_idx = randperm(num_sample);
%         X_train = train_X(re_idx,:);
%         Y_train = train_Y(re_idx);
        
        for j=1:num_sample
            if mod(j,1e3)==0
                toc;
                tic;
                fprintf('%d iter(%d epoch)---processing %dth sample\n', i, t, j);
            end

%             r = randi([1,num_sample]);
            if task == recommendation
                feature_idx = X_train(j,:);
                X = zeros(1, p);
                X(feature_idx) = 1;
                y = Y_train(j,:);
            else
                X = X_train(j,:);
                y = Y_train(j,:);
            end

            % pick anchor points
            [anchor_idx, weight] = knn(anchors, X, nearest_neighbor, beta);
            gamma = weight/sum(weight);

            y_anchor = zeros(1, nearest_neighbor);

            if task == recommendation
                for k=1:nearest_neighbor
                    temp_V = squeeze(V(feature_idx,:,anchor_idx(k)));
                    y_anchor(k) = sum(temp_V(1,:).*temp_V(2,:)) + w0(anchor_idx(k)) + sum(W(feature_idx,anchor_idx(k)));
                end
            else
                for k=1:nearest_neighbor                   
                    temp_V = V(:,:,anchor_idx(k));
                    tmp = sum(repmat(X',1,factors_num).*temp_V);
                    y_anchor(k) = (sum(tmp.^2) - sum(sum(repmat(X'.^2,1,factors_num).*(temp_V.^2))))/2 + w0(anchor_idx(k)) + X*W(:,anchor_idx(k));
                end
            end
           

            y_predict = gamma * y_anchor';

            if task == classification
                err = sigmf(y*y_predict,[1,0]);
            else
                err = y_predict - y;
            end

            idx = (t-1)*num_sample + j;
%             idx = j;
            if idx==1
                if task == classification
                    mse_da_llfm_sgd(idx) = -log(err);
                else
                    mse_da_llfm_sgd(idx) = err^2;
                end
            else
               if task == classification
                    mse_da_llfm_sgd(idx) = (mse_da_llfm_sgd(idx-1) * (idx - 1) -log(err))/idx;
                else
                    mse_da_llfm_sgd(idx) = (mse_da_llfm_sgd(idx-1) * (idx - 1) + err^2)/idx;
                end
            end

            % rmse_dallfm_train(i,t) = mse_da_llfm_sgd(idx);
            if task == classification
                rmse_dallfm_train(i, t) = mse_da_llfm_sgd(idx);
            else
                rmse_dallfm_train(i, t) = mse_da_llfm_sgd(idx)^0.5;
            end

            % update parameters
            if task == recommendation
                tmp_w0 = w0(anchor_idx);
                w0(anchor_idx) = tmp_w0 - learning_rate / (idx + t0) *  (2 * err * gamma);
                tmp_W = W(feature_idx,anchor_idx);
                W(feature_idx,anchor_idx) =  tmp_W - learning_rate / (idx + t0) *(2*err * repmat(gamma,2,1));

                for k=1:nearest_neighbor
                    temp_V = squeeze(V(feature_idx,:,anchor_idx(k)));
                   
                      V(feature_idx,:,anchor_idx(k)) = ...
                          temp_V - learning_rate / (idx + t0) * ...
                          (2*err*gamma(k)*(repmat(sum(temp_V),2,1)- temp_V));
                end

            end

            if task == classification
                tmp_w0 = w0(anchor_idx);
                w0(anchor_idx) = tmp_w0 - learning_rate / (idx + t0) * gamma .* (err-1)*y;
%                 w0(anchor_idx) = tmp_w0 - learning_rate * (gamma .* (err-1)*y + 2 * reg * tmp_w0);
                tmp_W = W(:,anchor_idx);
                W(:,anchor_idx) = tmp_W - learning_rate / (idx + t0) * repmat(gamma,p,1) .* ((err-1)*y*repmat(X',[1,nearest_neighbor]));
%                 W(:,anchor_idx) = tmp_W - learning_rate * (repmat(gamma,p,1) .* ((err-1)*y*repmat(X',[1,nearest_neighbor])) + 2 * reg * tmp_W);
                for k=1:nearest_neighbor
                    temp_V = squeeze(V(:,:,anchor_idx(k)));
                    V(:,:,anchor_idx(k)) = temp_V - learning_rate / (idx + t0) * gamma(k) * ((err-1)*y*(repmat(X',1,factors_num).*(repmat(X*temp_V,p,1)-repmat(X',1,factors_num).*temp_V)));
%                     V(:,:,anchor_idx(k)) = temp_V - learning_rate * (gamma(k) * ((err-1)*y*(repmat(X',1,factors_num).*(repmat(X*temp_V,p,1)-repmat(X',1,factors_num).*temp_V))) + 2 * reg * temp_V);
                end
            end

            if task == regression
                tmp_w0 = w0(anchor_idx);
                w0(anchor_idx) = tmp_w0 - learning_rate / (idx + t0) * gamma .* 2 * err;
                tmp_W = W(:,anchor_idx);
                W(:,anchor_idx) = tmp_W - learning_rate / (idx + t0) * repmat(gamma,p,1) .* (2*err*repmat(X',[1,nearest_neighbor]));
                for k=1:nearest_neighbor
                    temp_V = squeeze(V(:,:,anchor_idx(k)));
                    V(:,:,anchor_idx(k)) = temp_V - learning_rate / (idx + t0) * gamma(k) * (2*err*(repmat(X',1,factors_num).*(repmat(X*temp_V,p,1)-repmat(X',1,factors_num).*temp_V)));
                end
            end           
            % update anchor points
            
            count = count - 1;
            if count <= 0
                W = W * (1-skip/(idx+t0));
                V = V * (1-skip/(idx+t0));
                count = skip;
            end

            s = 2 * beta * (repmat(X, nearest_neighbor, 1) - anchors(anchor_idx, :)).*repmat(weight, p, 1)';
            base = -s * sum(weight.*y_anchor);
            base = base + repmat(y_anchor',1,p).* s*sum(weight);
%             anchors(anchor_idx,:) = anchors(anchor_idx,:) - learning_rate / (idx + t0) * (2*err* base/(sum(weight).^2));
            anchors(anchor_idx,:) = anchors(anchor_idx,:) - learning_rate / (idx + t0) * ((err-1)*y* base/(sum(weight).^2));

        end
    
        % validate
        mse_dallfm_test = 0.0;
        correct_num = 0;
        [num_sample_test, ~] = size(test_X);
        tic;
        for j=1:num_sample_test
            if mod(j,1000)==0
%                 toc;
%                 tic;
                fprintf('%d epoch(validation)---processing %dth sample\n',i, j);
             end

            if task == recommendation
                X = zeros(1, p);
                feature_idx = test_X(j,:);
                X(feature_idx) = 1;
                y = test_Y(j,:);
            else
                X = test_X(j,:);
                y = test_Y(j,:);
            end

            % pick anchor points
            [anchor_idx, weight] = knn(anchors, X, nearest_neighbor, beta);
            gamma = weight/sum(weight);

            y_anchor = zeros(1, nearest_neighbor);
            if task == recommendation
                for k=1:nearest_neighbor
                    temp_V = squeeze(V(feature_idx,:,anchor_idx(k)));
                    y_anchor(k) = sum(temp_V(1,:).*temp_V(2,:)) + w0(anchor_idx(k)) + sum(W(feature_idx,anchor_idx(k)));
                end
            else
                for k=1:nearest_neighbor
                    temp_V = squeeze(V(:,:,anchor_idx(k)));
                    tmp = sum(repmat(X',1,factors_num).*temp_V);
                    y_anchor(k) = (sum(tmp.^2) - sum(sum(repmat(X'.^2,1,factors_num).*(temp_V.^2))))/2 + w0(anchor_idx(k)) + X*W(:,anchor_idx(k));
                end
            end
            

            y_predict = gamma * y_anchor';

            if task == classification
                if (y_predict>=0 && y==1) || (y_predict<0&&y==-1)
                    correct_num = correct_num + 1;
                end
            end

            if task == classification
                err = sigmf(y*y_predict,[1,0]);
                mse_dallfm_test = mse_dallfm_test - log(err);
            else
                err = y_predict - y;
                mse_dallfm_test = mse_dallfm_test + err.^2;
            end
        end

        if task == classification
            accuracy_dallfm(i,t) = correct_num/num_sample_test;
        end
        
        if task == classification
            rmse_dallfm_test(i, t) = (mse_dallfm_test / num_sample_test);
        else
            rmse_dallfm_test(i, t) = (mse_dallfm_test / num_sample_test)^0.5;
        end 
        toc;
        fprintf('%d iter(%d epoch)---loss: %f\n', i,t,rmse_dallfm_test(i, t));
    end
end

%%
% validate

%%
% plot
plot(mse_da_llfm_sgd,'DisplayName','LLFMAAP');
legend('-DynamicLegend');
xlabel('Number of samples seen');
ylabel('RMSE');
hold on;
grid on;
%% 
plot(rmse_dallfm_test,'b--*','DisplayName','LLFM-APL');
legend('-DynamicLegend');
hold on;
% plot(rmse_dallfm_test,'DisplayName','DALLFM\_Test');
% legend('-DynamicLegend');
xlabel('epoch');
ylabel('RMSE');
% legend('DALLFM\_Train','DALLFM\_Test');
% title('DALLFM\_SGD');
grid on;
hold on;