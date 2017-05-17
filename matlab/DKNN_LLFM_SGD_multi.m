
rng('default');

class_num = max(train_Y);
[num_sample, p] = size(train_X);


% parameters 
iter_num = 1;

% ml 100k
learning_rate = 1e5;
t0 = 1e5;
skip = 1e3;
 
count = skip;

epoch = 10;

% locally linear
% anchor points
anchors_num = 50;

factors_num = 10;

% Lipschitz to noise ratio
% control the number of neighbours

LC = 0.5;
rmse_dadk_llfm_test = zeros(iter_num, epoch); 
rmse_dadk_llfm_train = zeros(iter_num, epoch); 

anchors_num_avg = zeros(1,iter_num);


bcon_dadkllfm = zeros(1,iter_num);
sumD_dadkllfm = zeros(1,iter_num);
accuracy_dadk_llfm = zeros(iter_num, epoch);


for i=1:iter_num
    
    % do shuffle
    re_idx = randperm(num_sample);
    X_train = train_X(re_idx,:);
    Y_train = train_Y(re_idx);

    num_nn = 0;
    num_nn_batch = 0;
    minmum_K = 100;
    maximum_K = 0;
    
    w0 = zeros(class_num, 1, anchors_num);
    W = zeros(class_num, p,anchors_num);
    V = 0.1*randn(class_num, p,factors_num,anchors_num);

    mse_dadk_llfm_sgd = zeros(1,num_sample);
    loss = zeros(1,num_sample);

    % get anchor points
    fprintf('Start K-means...\n');
    [~, anchors, ~, SD, ~] = litekmeans(train_X, anchors_num, 'Replicates', 10);

    fprintf('K-means done..\n');
    
    % SGD
    
    tic;
    for t=1:epoch
        
        num_nn_bach = 0;
        
        for j=1:num_sample
            if mod(j,1e3)==0
                toc;
                tic;

                fprintf('%d iter(%d epoch)---processing %dth sample\n', i, t, j);
                fprintf('batch average value of K in KNN is %.2f\n', num_nn_batch/1e3);
                fprintf('overall average value of K in KNN is %.2f\n', num_nn/((t-1)*num_sample+j));

                num_nn_batch = 0;

            end


            X = X_train(j,:);
            y = -ones(1, class_num);
            y(Y_train(j,:)) = 1;

            nz_idx = find(X);

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

            y_predict = zeros(1, class_num);
            y_anchor = zeros(class_num, nearest_neighbor);

            for u = 1:class_num
%                 y_anchor = zeros(1, nearest_neighbor);

                for k=1:nearest_neighbor               
                    temp_V = squeeze(V(u,nz_idx,:,anchor_idx(k)));
                    tmp = sum(repmat(X(nz_idx)',1,factors_num).*temp_V);
                    y_anchor(u,k) = (sum(tmp.^2) - sum(sum(repmat(X(nz_idx)'.^2,1,factors_num).*(temp_V.^2))))/2 + w0(u,anchor_idx(k)) + X(nz_idx)*squeeze(W(u,nz_idx,anchor_idx(k)))';
                end

                y_predict(u) = gamma * y_anchor(u,:)';
            end

            err = sigmf(y.*y_predict,[1,0]);

            idx = (t-1)*num_sample + j;

            if idx==1
                mse_dadk_llfm_sgd(idx) = sum(-log(err));
            else
                mse_dadk_llfm_sgd(idx) = (mse_dadk_llfm_sgd(idx-1) * (idx - 1) -sum(log(err)))/idx;
            end


            rmse_dadk_llfm_train(i, t) = mse_dadk_llfm_sgd(idx);

            % update parameters

            for u=1:class_num

                tmp_w0 = w0(u,anchor_idx);
                w0(u,anchor_idx) = tmp_w0 - learning_rate / (idx + t0) * (gamma .* (err(u)-1)*y(u));
                tmp_W = squeeze(W(u,nz_idx,anchor_idx));
                if nearest_neighbor == 1
                    W_ = learning_rate / (idx + t0) * ((err(u)-1)*repmat(gamma,length(nz_idx),1)*y(u).*repmat(X(nz_idx)',[1,nearest_neighbor]))';
                else
                    W_ = learning_rate / (idx + t0) * ((err(u)-1)*repmat(gamma,length(nz_idx),1)*y(u).*repmat(X(nz_idx)',[1,nearest_neighbor]));
                end
                W(u,nz_idx,anchor_idx) = tmp_W - W_; 
                for k=1:nearest_neighbor
                    temp_V = squeeze(V(u,:,:,anchor_idx(k)));
                    V(u,nz_idx,:,anchor_idx(k)) = temp_V -...
                     learning_rate / (idx + t0) *...
                      ((err(u)-1)*gamma(k)*y(u)*(repmat(X(nz_idx)',1,factors_num).*(repmat(X(nz_idx)*temp_V,length(nz_idx),1)-repmat(X(nz_idx)',1,factors_num).*temp_V)));
                end
                
                % update anchor points

                s = 2 * LC * (repmat(X, nearest_neighbor, 1) - anchors(anchor_idx, :));
                base = -s * sum(weight.*y_anchor(u,:));
                base = base + repmat(y_anchor(u,:)',1,p).* s*sum(weight);
                anchors(anchor_idx,:) = anchors(anchor_idx,:) - learning_rate / 100/ (idx + t0) * ((err(u)-1)*y(u)* base/(sum(weight).^2));

            end

            count = count - 1;
            if count <= 0
                W = W * (1-skip/(idx+t0));
                V = V * (1-skip/(idx+t0));
                count = skip;
            end


            % s = 2 * LC * (repmat(X, nearest_neighbor, 1) - anchors(anchor_idx, :));
            % base = -s * sum(weight.*y_anchor);
            % base = base + repmat(y_anchor',1,p).* s*sum(weight);
            % anchors(anchor_idx,:) = anchors(anchor_idx,:) - learning_rate / (idx + t0) * (err-1) * y * base/(sum(weight).^2);

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
                fprintf('%d epoch(validation)---processing %dth sample\n',t, j);
            end

            X = test_X(j,:);
            y = -ones(1, class_num);
            y(test_Y(j,:)) = 1;
            
            nz_idx = find(X);

            [anchor_idx, D, lam] = Dynamic_KNN(anchors, X, LC);
            nearest_neighbor = length(anchor_idx);

            weight = lam - D;

            gamma = weight/sum(weight);

            y_predict = zeros(1, class_num);

            for u = 1:class_num
                y_anchor = zeros(1, nearest_neighbor);

                for k=1:nearest_neighbor               
                    temp_V = squeeze(V(u,nz_idx,:,anchor_idx(k)));
                    tmp = sum(repmat(X(nz_idx)',1,factors_num).*temp_V);
                    y_anchor(k) = (sum(tmp.^2) - sum(sum(repmat(X(nz_idx)'.^2,1,factors_num).*(temp_V.^2))))/2 + w0(u,anchor_idx(k)) + X(nz_idx)*squeeze(W(u,nz_idx,anchor_idx(k)))';
                end

                y_predict(u) = gamma * y_anchor';
            end

            err = sigmf(y.*y_predict,[1,0]);
            mse_dadk_llfm_test = mse_dadk_llfm_test - sum(log(err));

            [~, label] = max(y_predict);
            % accuracy
            if label == test_Y(j,:)
                correct_num = correct_num + 1;
            end

        end

        accuracy_dadk_llfm(i,t) = correct_num/num_sample_test;

        rmse_dadk_llfm_test(i, t) = (mse_dadk_llfm_test / num_sample_test);

        toc;
        fprintf('%d iter(%d epoch)---loss: %f\t accuracy: %f\n', i,t,rmse_dadk_llfm_test(i, t),accuracy_dadk_llfm(i,t));
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
plot(rmse_dadk_llfm_test,'r--x', 'DisplayName','LLFM-JO');
legend('-DynamicLegend');
hold on;
% plot(rmse_dadk_llfm_test,'DisplayName','DKDKLLFM\_Test');
% legend('-DynamicLegend');
xlabel('epoch');
ylabel('RMSE');
% legend('DKDALLFM\_Train','DKDKLLFM\_Test');
% title('DADKLLFM\_SGD');
grid on;
hold on;