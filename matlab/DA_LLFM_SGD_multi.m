 
rng('default');
   
[num_sample, p] = size(train_X);

% parameters 
iter_num = 1;
epoch = 20;

factors_num = 10;

learning_rate = 1e4;
t0 = 1e5;
skip = 1e1;


count = skip;

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
    [~, anchors, ~, SD, ~] = litekmeans(train_X, anchors_num, 'Replicates', 10);
    fprintf('K-means done..\n');
    
    % SGD
    tic;
    % do shuffle
    re_idx = randperm(num_sample);
    X_train = train_X(re_idx,:);
    Y_train = train_Y(re_idx);
    
    for t=1:epoch

        for j=1:num_sample
            if mod(j,1e3)==0
                toc;
                tic;
                fprintf('%d iter(%d epoch)---processing %dth sample\n', i, t, j);
            end


            X = X_train(j,:);
            y = Y_train(j,:);

            % pick anchor points
            [anchor_idx, weight] = knn(anchors, X, nearest_neighbor, beta);
            gamma = weight/sum(weight);

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
                mse_da_llfm_sgd(idx) = -log(err);
            else
                mse_da_llfm_sgd(idx) = (mse_da_llfm_sgd(idx-1) * (idx - 1) -log(err))/idx;
            end

            rmse_dallfm_train(i, t) = mse_da_llfm_sgd(idx);


            % update parameters

            if task == classification
                tmp_w0 = w0(anchor_idx);
                w0(anchor_idx) = tmp_w0 - learning_rate / (idx + t0) * gamma .* (err-1)*y;
                tmp_W = W(:,anchor_idx);
                W(:,anchor_idx) = tmp_W - learning_rate / (idx + t0) * repmat(gamma,p,1) .* ((err-1)*y*repmat(X',[1,nearest_neighbor]));
                for k=1:nearest_neighbor
                    temp_V = squeeze(V(:,:,anchor_idx(k)));
                    V(:,:,anchor_idx(k)) = temp_V - learning_rate / (idx + t0) * gamma(k) * ((err-1)*y*(repmat(X',1,factors_num).*(repmat(X*temp_V,p,1)-repmat(X',1,factors_num).*temp_V)));
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
            anchors(anchor_idx,:) = anchors(anchor_idx,:) - learning_rate / (idx + t0) * ((err-1)*y* base/(sum(weight).^2));

        end
    
        % validate
        mse_dallfm_test = 0.0;
        correct_num = 0;
        [num_sample_test, ~] = size(test_X);
        tic;
        for j=1:num_sample_test
            if mod(j,1000)==0
                toc;
                tic;
                fprintf('%d epoch(validation)---processing %dth sample\n',, j);
             end

            X = test_X(j,:);
            y = test_Y(j,:);

            % pick anchor points
            [anchor_idx, weight] = knn(anchors, X, nearest_neighbor, beta);
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
            mse_dallfm_test = mse_dallfm_test - log(err);
        end


        accuracy_dallfm(i,t) = correct_num/num_sample_test;

        rmse_dallfm_test(i, t) = (mse_dallfm_test / num_sample_test);

        toc;
        fprintf('%d iter(%d epoch)---loss: %f\t accuracy: %f\n', i,t,rmse_dallfm_test(i, t),accuracy_dallfm(i,t));
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