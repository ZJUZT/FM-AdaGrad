% load training data
% train_X, train_Y
% load('training_data_100k');
% load('test_data_100k');
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
% p = max(train_X(:,2));

% y_max = max(train_Y);
% y_min = min(train_Y);

% parameters  
iter_num = 1 ;
factors_num = 10;

% ml 100k
% learning_rate = 1e4;
% t0 = 1e4;
% skip = 1e3;

% netflix

% banana
learning_rate = 1e4;
t0 = 1e5;
skip = 1e2; 

% ijcnn
% learning_rate = 1e5 ;
% t0 = 1e5;
% skip = 1e3;   

% ml 100k
% learning_rate = 2e4;
% t0 = 1e5;
% skip = 1e3;   

% learning_rate = 1e-1;
% reg = 1e-5;

% netflix
% learning_rate = 1e5;  
% t0 = 1e5;
% skip = 1e3; 

% magic04
% learning_rate = 5e4;
% t0 = 1e5;
% skip = 1e3; 

count = skip;

% locally linear
% anchor points
anchors_num = 10 ;


epoch = 50;

% knn
nearest_neighbor = 5  ;
beta = 1;

bcon_llfm = zeros(1,iter_num);
sumD_llfm = zeros(1,iter_num);



rmse_llfm_test = zeros(iter_num, epoch);
rmse_llfm_train = zeros(iter_num, epoch);
accuracy_llfm = zeros(iter_num, epoch);

% random pick
% idx = randperm(num_sample);
% % anchors = train_X(idx(1:anchors_num), :);
% anchors = 0.01* rand(anchors_num, p);

% T = 1e5;

for i=1:iter_num
    % do shuffle
%     re_idx = randperm(num_sample);
%     X_train = train_X(re_idx,:);
%     Y_train = train_Y(re_idx);
    
    w0 = zeros(1, anchors_num);
    W = zeros(p,anchors_num);
    V = 0.1*randn(p,factors_num,anchors_num);

    % initialization by pre-FM-train
%     w0 = repmat(w0_ini, 1, anchors_num);
% %     W = zeros(p,anchors_num);
%     W = repmat(W_ini', 1, anchors_num);
% %     V = 0.1*randn(p,factors_num,anchors_num);
%     V = repmat(V_ini, 1,1, anchors_num);

    mse_llfm_sgd = zeros(1,num_sample);
    loss = zeros(1,epoch*num_sample);
    
    

    % get anchor points
    fprintf('Start K-means...\n');
%     [label, anchors, bcon_llfm(i), SD, ~] = litekmeans(sparse_matrix(train_X), anchors_num, 'Replicates', 10);
    [~, anchors, bcon_llfm(i), SD, ~] = litekmeans(train_X, anchors_num, 'Replicates', 10);
%     sumD_llfm(i) = sum(SD);

%     [label, anchors, bcon_llfm(i), SD, ~] = litekmeans(train_X, anchors_num);
    
%     anchors = 0.1* randn(anchors_num, p);
    fprintf('K-means done..\n');
    
    % SGD
    tic;
    
    for t=1:epoch
        
%         re_idx = randperm(num_sample);
%         X_train = train_X(re_idx,:);
%         Y_train = train_Y(re_idx);
%         X_train = train_X;
%         Y_train = train_Y;

        re_idx = randperm(num_sample);
        X_train = train_X(re_idx,:);
        Y_train = train_Y(re_idx);
        
        for j=1:num_sample
            if mod(j,1e3)==0
                toc;
                tic;
                fprintf('%d iter(%d epoch)---processing %dth sample\n', i,t,j);
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

    %         temp_V = V(:,:,anchor_idx);
    %         X_ = repmat(X', [1, factors_num, nearest_neighbor]);
    %         tmp = X_.*temp_V;
    %         factor_part = gamma * (squeeze(sum(sum(tmp.^2,1),2) -(sum(sum((X_.^2).*(temp_V.*2),1),2))));

            y_predict = gamma * y_anchor';
            
%             y_predict = min(y_predict, y_max);
%             y_predict = max(y_predict, y_min);

            if task == classification
                err = sigmf(y*y_predict,[1,0]);
            else
                err = y_predict - y;
            end

            idx = (t-1)*num_sample + j;
    %         loss(idx) = err^2;
    %         mse_llfm_sgd(idx) = sum(loss)/idx;
%             idx = (t-1)*num_sample + j;
%             idx = j;
            if idx==1
                if task == classification
                    mse_llfm_sgd(idx) = -log(err);
                else
                    mse_llfm_sgd(idx) = err^2;
                end
            else
               if task == classification
                    mse_llfm_sgd(idx) = (mse_llfm_sgd(idx-1) * (idx - 1) -log(err))/idx;
                else
                    mse_llfm_sgd(idx) = (mse_llfm_sgd(idx-1) * (idx - 1) + err^2)/idx;
                end
            end

            if task == classification
                rmse_llfm_train(i, t) = mse_llfm_sgd(idx);
            else
                rmse_llfm_train(i, t) = mse_llfm_sgd(idx)^0.5;
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
                w0(anchor_idx) = tmp_w0 - learning_rate / (idx + t0) * (gamma .* (err-1)*y);
%                 w0(anchor_idx) = tmp_w0 - learning_rate  * (gamma .* (err-1)*y + 2 * reg * w0(anchor_idx));
                tmp_W = W(:,anchor_idx);
                W(:,anchor_idx) = tmp_W - learning_rate / (idx + t0) * ((err-1)*repmat(gamma,p,1)*y.*repmat(X',[1,nearest_neighbor]));
%                 W(:,anchor_idx) = tmp_W - learning_rate * ((err-1)*repmat(gamma,p,1)*y.*repmat(X',[1,nearest_neighbor]) + 2 * reg * tmp_W);
                for k=1:nearest_neighbor
                    temp_V = squeeze(V(:,:,anchor_idx(k)));
                    V(:,:,anchor_idx(k)) = temp_V - learning_rate / (idx + t0) * ((err-1)*gamma(k)*y*(repmat(X',1,factors_num).*(repmat(X*temp_V,p,1)-repmat(X',1,factors_num).*temp_V)));
%                     V(:,:,anchor_idx(k)) = temp_V - learning_rate * ((err-1)*gamma(k)*y*(repmat(X',1,factors_num).*(repmat(X*temp_V,p,1)-repmat(X',1,factors_num).*temp_V)) + 2 * reg * temp_V);
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
            
            count = count - 1;
            if count <= 0
                W = W * (1-skip/(idx+t0));
                V = V * (1-skip/(idx+t0));
                count = skip;
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
                mse_llfm_test = mse_llfm_test - log(err);
            else
                err = y_predict - y;
                mse_llfm_test = mse_llfm_test + err.^2;
            end
        end

        if task == classification
            accuracy_llfm(i,t) = correct_num/num_sample_test;
        end
        
        if task == classification
            rmse_llfm_test(i, t) = (mse_llfm_test / num_sample_test);
        else
            rmse_llfm_test(i, t) = (mse_llfm_test / num_sample_test)^0.5;
        end
        
        toc;
        fprintf('%d iter(%d epoch)---loss: %f\n', i,t,rmse_llfm_test(i, t));
    end
end

%%


%%
% plot
plot(mse_llfm_sgd,'DisplayName','LLFM');
legend('-DynamicLegend');
xlabel('Number of samples seen');
ylabel('RMSE'); 
grid on;
hold on;

%%
plot(rmse_llfm_test,'g--+','DisplayName','LLFM-DO');
legend('-DynamicLegend');
hold on;
% plot(rmse_llfm_test,'DisplayName','LLFM\_Test');
% legend('-DynamicLegend');
xlabel('epoch');
ylabel('RMSE');
% legend('LLFM\_Train','LLFM\_Test');
% title('LLFM\_SGD');
grid on;

%%
% X_ = zeros(num_sample,p);
% for i=1:num_sample
%     X_(i,X_train(i,:)) = 1;
% end
    