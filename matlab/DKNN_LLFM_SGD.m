% load training data
% train_X, train_Y

% fully adaptive (KNN && anchor points)

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

% parameters 
iter_num = 5;

% banana
% learning_rate = 1e4;
% t0 = 1e5;
% skip = 1e3;

% ijcnn
% learning_rate = 1e5;
% t0 = 1e5;
% skip = 1e3;

% banana
% learning_rate = 1e5;
% t0 = 1e5;
% skip = 1e3;

% magic
learning_rate = 1e4;
t0 = 1e5;
skip = 1e3;


count = skip;

epoch = 15;

% locally linear
% anchor points
anchors_num = 50;

factors_num = 10;

% Lipschitz to noise ratio
% control the number of neighbours

% ml 100
% LC = 0.08;

% ijcnn
% LC = 1;

% magic
LC = 1;

%banana
% LC = 1;
rmse_dadk_llfm_test = zeros(iter_num, epoch); 
rmse_dadk_llfm_train = zeros(iter_num, epoch); 

anchors_num_avg = zeros(1,iter_num);

% knn
% nearest_neighbor = 10;

% T = 1e5;

bcon_dadkllfm = zeros(1,iter_num);
sumD_dadkllfm = zeros(1,iter_num);
accuracy_dadk_llfm = zeros(iter_num, epoch);


% initial anchor points
% [~, anchors, ~, ~, ~] = litekmeans(train_X, anchors_num);
% idx = randperm(num_sample);
% anchors = train_X(idx(1:anchors_num), :);
% anchors = 0.01*rand(anchors_num, p);


for i=1:iter_num
    
    
    num_nn = 0;
    num_nn_batch = 0;
    minmum_K = 100;
    maximum_K = 0;
    
    w0 = zeros(1, anchors_num);
    W = zeros(p,anchors_num);
    V = 0.1*randn(p,factors_num,anchors_num);

    mse_dadk_llfm_sgd = zeros(1,num_sample);
    loss = zeros(1,num_sample);

    % get anchor points
    fprintf('Start K-means...\n');
%     [~, anchors, bcon_dadkllfm(i), SD, ~] = litekmeans(sparse_matrix(train_X), anchors_num, 'Replicates', 10);
    [~, anchors, ~, SD, ~] = litekmeans(train_X, anchors_num);
%     sumD_dadkllfm(i) = sum(SD);
%     anchors = 0.1*randn(anchors_num, p);

    
    fprintf('K-means done..\n');
    
    % SGD
    
    % do shuffle
    re_idx = randperm(num_sample);
    X_train = train_X(re_idx,:);
    Y_train = train_Y(re_idx);
    
    tic;
    for t=1:epoch
        
        num_nn_bach = 0;
%         X_train = train_X;
%         Y_train = train_Y;

%         do shuffle
%         re_idx = randperm(num_sample);
%         X_train = train_X(re_idx,:);
%         Y_train = train_Y(re_idx);
        
        for j=1:num_sample
            if mod(j,1e3)==0
                toc;
                tic;

                fprintf('%d iter(%d epoch)---processing %dth sample\n', i, t, j);
                fprintf('batch average value of K in KNN is %.2f\n', num_nn_batch/1e3);
                fprintf('overall average value of K in KNN is %.2f\n', num_nn/((t-1)*num_sample+j));

                num_nn_batch = 0;

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
                
                nz_idx = find(X);
            end

            [anchor_idx, D, lam] = Dynamic_KNN(anchors, X, LC);
            nearest_neighbor = length(anchor_idx);
            if minmum_K>nearest_neighbor
                minmum_K=nearest_neighbor;
            end

            if maximum_K<nearest_neighbor
                maximum_K=nearest_neighbor;
            end

            weight = lam - D;

            gamma = weight/sum(weight);



            num_nn = num_nn + nearest_neighbor;
            num_nn_batch = num_nn_batch + nearest_neighbor;

            y_anchor = zeros(1, nearest_neighbor);

            if task == recommendation
                for k=1:nearest_neighbor
                    temp_V = squeeze(V(feature_idx,:,anchor_idx(k)));
                    y_anchor(k) = sum(temp_V(1,:).*temp_V(2,:)) + w0(anchor_idx(k)) + sum(W(feature_idx,anchor_idx(k)));
                end
            else
                for k=1:nearest_neighbor                   
                    temp_V = V(nz_idx,:,anchor_idx(k));
                    tmp = sum(repmat(X(nz_idx)',1,factors_num).*temp_V);
                    y_anchor(k) = (sum(tmp.^2) - sum(sum(repmat(X(nz_idx)'.^2,1,factors_num).*(temp_V.^2))))/2 + w0(anchor_idx(k)) + X(nz_idx)*W(nz_idx,anchor_idx(k));
                end
            end


            y_predict = gamma * y_anchor';

            if task == classification
                err = sigmf(y*y_predict,[1,0]);
            else
                err = y_predict - y;
            end


%             idx = j;
            idx = (t-1)*num_sample + j;
            if idx==1
                if task == classification
                    mse_dadk_llfm_sgd(idx) = -log(err);
                else
                    mse_dadk_llfm_sgd(idx) = err^2;
                end
            else
               if task == classification
                    mse_dadk_llfm_sgd(idx) = (mse_dadk_llfm_sgd(idx-1) * (idx - 1) -log(err))/idx;
                else
                    mse_dadk_llfm_sgd(idx) = (mse_dadk_llfm_sgd(idx-1) * (idx - 1) + err^2)/idx;
                end
            end

            if task == classification
                rmse_dadk_llfm_train(i, t) = mse_dadk_llfm_sgd(idx);
            else
                rmse_dadk_llfm_train(i, t) = mse_dadk_llfm_sgd(idx)^0.5;
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
                tmp_W = W(nz_idx,anchor_idx);
                W_ = learning_rate / (idx + t0) * repmat(gamma,length(nz_idx),1) .* ((err-1)*y*repmat(X(nz_idx)',[1,nearest_neighbor]));
                W(nz_idx,anchor_idx) = tmp_W - W_;
                for k=1:nearest_neighbor
                    temp_V = squeeze(V(nz_idx,:,anchor_idx(k)));
                    V(nz_idx,:,anchor_idx(k)) = temp_V - learning_rate / (idx + t0) * gamma(k) * ((err-1)*y*(repmat(X(nz_idx)',1,factors_num).*(repmat(X(nz_idx)*temp_V,length(nz_idx),1)-repmat(X(nz_idx)',1,factors_num).*temp_V)));
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

            % tmp_w0 = w0(anchor_idx);
            % w0(anchor_idx) = tmp_w0 - learning_rate * gamma .* (err-1)*y;
            % tmp_W = W(:,anchor_idx);
            % W(:,anchor_idx) = tmp_W - learning_rate * repmat(gamma,p,1) .* ((err-1)*y*repmat(X',[1,nearest_neighbor]) + 2*reg_w*tmp_W);

            % tmp_W = W(feature_idx,anchor_idx);
            % W(feature_idx,anchor_idx) =  tmp_W - learning_rate * (2*err * repmat(gamma,2,1) + 2*reg_w*tmp_W);


    %         s = 2 * LC * (repmat(X, nearest_neighbor, 1) - anchors(anchor_idx, :));
    %         base = -s .* repmat(weight, p, 1)';

    %         s = 2 * LC * (repmat(X, nearest_neighbor, 1) - anchors(anchor_idx, :)).*repmat(weight, p, 1)';

    %         s = repmat(-2*LC*(sum(weight.^2)-...
    %             nearest_neighbor*weight)/sqrt(nearest_neighbor+...
    %             sum(weight).^2-nearest_neighbor*sum(weight.^2)),p,1)' .* ...
    %             (repmat(X, nearest_neighbor, 1) - anchors(anchor_idx, :));
    %         s1 = repmat(2*LC-2*LC/nearest_neighbor-2*LC*(sum(weight.^2)-...
    %             nearest_neighbor*weight)/nearest_neighbor/sqrt(nearest_neighbor+...
    %             sum(weight).^2-nearest_neighbor*sum(weight.^2)),p,1)' .* ...
    %             (repmat(X, nearest_neighbor, 1) - anchors(anchor_idx, :));

    %         base = -s .* repmat(weight, p, 1)';
    %         base = repmat(y_anchor * base,nearest_neighbor,1) + repmat(y_anchor',1,p).* s*sum(weight);

            s = 2 * LC * (repmat(X, nearest_neighbor, 1) - anchors(anchor_idx, :));
            base = -s * sum(weight.*y_anchor);
            base = base + repmat(y_anchor',1,p).* s*sum(weight);
%             anchors(anchor_idx,:) = anchors(anchor_idx,:) - learning_rate / (idx + t0) * 2 * err * base/(sum(weight).^2);
            anchors(anchor_idx,:) = anchors(anchor_idx,:) - learning_rate / 1e1 / (idx + t0) * (err-1) * y * base/(sum(weight).^2);


    %         for k=1:nearest_neighbor
    % %             temp_V = squeeze(V(:,:,anchor_idx(k)));
    % %             V(:,:,anchor_idx(k)) = temp_V - learning_rate * gamma(k) * ...
    % %                 (2*err*(repmat(X',1,factors_num).*(repmat(X*temp_V,p,1)-repmat(X',1,factors_num).*temp_V)) + 2*reg_v*squeeze(temp_V));
    %             % temp_V = squeeze(V(feature_idx,:,anchor_idx(k)));

    %             % V(feature_idx,:,anchor_idx(k)) = ...
    %             %       temp_V - learning_rate * ...
    %             %       (2*err*gamma(k)*((repmat(sum(temp_V),2,1))- temp_V) + 2*reg_v*temp_V);
    %             temp_V = squeeze(V(:,:,anchor_idx(k)));
    %             V(:,:,anchor_idx(k)) = temp_V - learning_rate * gamma(k) * ((err-1)*y*(repmat(X',1,factors_num).*(repmat(X*temp_V,p,1)-repmat(X',1,factors_num).*temp_V)) + 2*reg_v*squeeze(temp_V));

    %             % update anchor points
    % %             tmp = anchors(anchor_idx(k), :);
    % %             delt = base;
    % %             delt(k, :) = delt(k,:) + s(k,:) * sum(weight);
    % %             delt = delt / (sum(weight).^2);
    % %             
    % %             anchors(anchor_idx(k), :) = tmp - learning_rate_anchor * 2 * err * y_anchor*delt;
    %         end

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

        anchors_num_avg(i) = num_nn/num_sample;

        % validate
        mse_dadk_llfm_test = 0.0;
        correct_num = 0;
        [num_sample_test, ~] = size(test_X);
        tic;
        for j=1:num_sample_test
            if mod(j,1000)==0
                toc;
                tic;
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
                nz_idx = find(X);
            end

            % X = zeros(1, p);
            % feature_idx = test_X(k,:);
            % X(feature_idx) = 1;
            % y = test_Y(k,:);

            % pick anchor points
    %         [anchor_idx, weight] = knn(anchors, X, nearest_neighbor);

            [anchor_idx, D, lam] = Dynamic_KNN(anchors, X, LC);
            nearest_neighbor = length(anchor_idx);

            weight = lam - D;

            gamma = weight/sum(weight);

            y_anchor = zeros(1, nearest_neighbor);
    %         for n=1:nearest_neighbor
    %             temp_V = squeeze(V(:,:,anchor_idx(n)));
    %             tmp = sum(repmat(X',1,factors_num).*temp_V);
    %             y_anchor(n) = (sum(tmp.^2) - sum(sum(repmat(X'.^2,1,factors_num).*(temp_V.^2))))/2 + w0(anchor_idx(n)) + X*W(:,anchor_idx(n));
    %         end

            if task == recommendation
                for k=1:nearest_neighbor
                    temp_V = squeeze(V(feature_idx,:,anchor_idx(k)));
                    y_anchor(k) = sum(temp_V(1,:).*temp_V(2,:)) + w0(anchor_idx(k)) + sum(W(feature_idx,anchor_idx(k)));
                end
            else
                for k=1:nearest_neighbor
                    temp_V = squeeze(V(nz_idx,:,anchor_idx(k)));
                    tmp = sum(repmat(X(nz_idx)',1,factors_num).*temp_V);
                    y_anchor(k) = (sum(tmp.^2) - sum(sum(repmat(X(nz_idx)'.^2,1,factors_num).*(temp_V.^2))))/2 + w0(anchor_idx(k)) + X(nz_idx)*W(nz_idx,anchor_idx(k));
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
                mse_dadk_llfm_test = mse_dadk_llfm_test - log(err);
            else
                err = y_predict - y;
                mse_dadk_llfm_test = mse_dadk_llfm_test + err.^2;
            end

        end

        if task == classification
            accuracy_dadk_llfm(i,t) = correct_num/num_sample_test;
        end
        
        if task == classification
            rmse_dadk_llfm_test(i, t) = (mse_dadk_llfm_test / num_sample_test);
        else
            rmse_dadk_llfm_test(i, t) = (mse_dadk_llfm_test / num_sample_test)^0.5;
        end
        toc;
        fprintf('%d iter(%d epoch)---loss: %f\n', i,t,rmse_dadk_llfm_test(i, t));
    end
end

%%
% validate

%%
% plot
plot(mse_dadk_llfm_sgd,'DisplayName','LLFMAAPK');
legend('-DynamicLegend');
xlabel('Number of samples seen');
ylabel('logloss');
hold on;
grid on;

%%
plot(rmse_dadk_llfm_test(1,:),'r--x', 'DisplayName','LLFM-JO');
legend('-DynamicLegend');
hold on;
% plot(rmse_dadk_llfm_test,'DisplayName','DKDKLLFM\_Test');
% legend('-DynamicLegend');
xlabel('epoch');
ylabel('logloss');
% legend('DKDALLFM\_Train','DKDKLLFM\_Test');
% title('DADKLLFM\_SGD');
grid on;
hold on;