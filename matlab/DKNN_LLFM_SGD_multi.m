
rng('default');

[num_sample, p] = size(train_X);

% parameters 
iter_num = 1;

% ml 100k
learning_rate = 1e4;
t0 = 1e5;
skip = 1e3;
 
count = skip;

epoch = 20;

% locally linear
% anchor points
anchors_num = 10;

factors_num = 10;

% Lipschitz to noise ratio
% control the number of neighbours

LC = 1;
rmse_dadk_llfm_test = zeros(iter_num, epoch); 
rmse_dadk_llfm_train = zeros(iter_num, epoch); 

anchors_num_avg = zeros(1,iter_num);


bcon_dadkllfm = zeros(1,iter_num);
sumD_dadkllfm = zeros(1,iter_num);
accuracy_dadk_llfm = zeros(iter_num, epoch);


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
    [~, anchors, ~, SD, ~] = litekmeans(train_X, anchors_num, 'Replicates', 10);

    fprintf('K-means done..\n');
    
    % SGD
    
    % do shuffle
    re_idx = randperm(num_sample);
    X_train = train_X(re_idx,:);
    Y_train = train_Y(re_idx);
    
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
            y = Y_train(j,:);

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


            for k=1:nearest_neighbor                   
                temp_V = V(:,:,anchor_idx(k));
                tmp = sum(repmat(X',1,factors_num).*temp_V);
                y_anchor(k) = (sum(tmp.^2) - sum(sum(repmat(X'.^2,1,factors_num).*(temp_V.^2))))/2 + w0(anchor_idx(k)) + X*W(:,anchor_idx(k));
            end

            y_predict = gamma * y_anchor';

            err = sigmf(y*y_predict,[1,0]);

            idx = (t-1)*num_sample + j;
            if idx==1
                mse_dadk_llfm_sgd(idx) = -log(err);
            else
                mse_dadk_llfm_sgd(idx) = (mse_dadk_llfm_sgd(idx-1) * (idx - 1) -log(err))/idx;
            end


            rmse_dadk_llfm_train(i, t) = mse_dadk_llfm_sgd(idx);

            % update parameters

            tmp_w0 = w0(anchor_idx);
            w0(anchor_idx) = tmp_w0 - learning_rate / (idx + t0) * gamma .* (err-1)*y;
            tmp_W = W(:,anchor_idx);
            W(:,anchor_idx) = tmp_W - learning_rate / (idx + t0) * repmat(gamma,p,1) .* ((err-1)*y*repmat(X',[1,nearest_neighbor]));
            for k=1:nearest_neighbor
                temp_V = squeeze(V(:,:,anchor_idx(k)));
                V(:,:,anchor_idx(k)) = temp_V - learning_rate / (idx + t0) * gamma(k) * ((err-1)*y*(repmat(X',1,factors_num).*(repmat(X*temp_V,p,1)-repmat(X',1,factors_num).*temp_V)));
            end

            count = count - 1;
            if count <= 0
                W = W * (1-skip/(idx+t0));
                V = V * (1-skip/(idx+t0));
                count = skip;
            end


            s = 2 * LC * (repmat(X, nearest_neighbor, 1) - anchors(anchor_idx, :));
            base = -s * sum(weight.*y_anchor);
            base = base + repmat(y_anchor',1,p).* s*sum(weight);
            anchors(anchor_idx,:) = anchors(anchor_idx,:) - learning_rate / (idx + t0) * (err-1) * y * base/(sum(weight).^2);

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
            y = test_Y(j,:);

            [anchor_idx, D, lam] = Dynamic_KNN(anchors, X, LC);
            nearest_neighbor = length(anchor_idx);

            weight = lam - D;

            gamma = weight/sum(weight);

            y_anchor = zeros(1, nearest_neighbor);

            for k=1:nearest_neighbor
                temp_V = squeeze(V(:,:,anchor_idx(k)));
                tmp = sum(repmat(X',1,factors_num).*temp_V);
                y_anchor(k) = (sum(tmp.^2) - sum(sum(repmat(X'.^2,1,factors_num).*(temp_V.^2))))/2 + w0(anchor_idx(k)) + X*W(:,anchor_idx(k));
            end

            y_predict = gamma * y_anchor';

            if (y_predict>=0 && y==1) || (y_predict<0&&y==-1)
                correct_num = correct_num + 1;
            end

            err = sigmf(y*y_predict,[1,0]);
            mse_dadk_llfm_test = mse_dadk_llfm_test - log(err);

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