
rng('default');

class_num = max(train_Y);
[num_sample, p] = size(train_X);

% parameters  
iter_num = 1 ;
factors_num = 10;

learning_rate = 1e5;
t0 = 1e5;
skip = 1e3 ; 

count = skip;

% locally linear
% anchor points
anchors_num = 50 ;


epoch = 10;

% knn
nearest_neighbor = 5;
beta = 1;

bcon_llfm = zeros(1,iter_num);
sumD_llfm = zeros(1,iter_num);

rmse_llfm_test = zeros(iter_num, epoch);
rmse_llfm_train = zeros(iter_num, epoch);
accuracy_llfm = zeros(iter_num, epoch);


for i=1:iter_num
    % do shuffle
    re_idx = randperm(num_sample);
    X_train = train_X(re_idx,:);
    Y_train = train_Y(re_idx);
    
    w0 = zeros(class_num, 1, anchors_num);
    W = zeros(class_num, p,anchors_num);

    % 4-dimensional ?? freaking kidding me ???
    V = 0.1*randn(class_num, p,factors_num,anchors_num);

    mse_llfm_sgd = zeros(1,num_sample);
    loss = zeros(1,epoch*num_sample);

    % get anchor points
    fprintf('Start K-means...\n');

    [~, anchors, bcon_llfm(i), SD, ~] = litekmeans(train_X, anchors_num, 'Replicates', 10);

    fprintf('K-means done..\n');
    
    % SGD
    tic;
    
    for t=1:epoch
        
        for j=1:num_sample
            if mod(j,1e3)==0
                toc;
                tic;
                fprintf('%d iter(%d epoch)---processing %dth sample\n', i,t,j);
            end


            X = X_train(j,:);
            y = -ones(1, class_num);
            y(Y_train(j,:)) = 1;

            nz_idx = find(X);

            % pick anchor points
            [anchor_idx, weight] = knn(anchors, X, nearest_neighbor, beta);         
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


            idx = (t-1)*num_sample + j;

            if idx==1
                mse_llfm_sgd(idx) = sum(-log(err));
            else
                mse_llfm_sgd(idx) = (mse_llfm_sgd(idx-1) * (idx - 1) - sum(log(err)))/idx;
            end

            rmse_llfm_train(i, t) = mse_llfm_sgd(idx);

            % update parameters

            for u=1:class_num

                tmp_w0 = w0(u,anchor_idx);
                w0(u,anchor_idx) = tmp_w0 - learning_rate / (idx + t0) * (gamma .* (err(u)-1)*y(u));
                tmp_W = squeeze(W(u,nz_idx,anchor_idx));
                W(u,nz_idx,anchor_idx) = tmp_W - learning_rate / (idx + t0) * ((err(u)-1)*repmat(gamma,length(nz_idx),1)*y(u).*repmat(X(nz_idx)',[1,nearest_neighbor]));
                for k=1:nearest_neighbor
                    temp_V = squeeze(V(u,:,:,anchor_idx(k)));
                    V(u,nz_idx,:,anchor_idx(k)) = temp_V -...
                     learning_rate / (idx + t0) *...
                      ((err(u)-1)*gamma(k)*y(u)*(repmat(X(nz_idx)',1,factors_num).*(repmat(X(nz_idx)*temp_V,length(nz_idx),1)-repmat(X(nz_idx)',1,factors_num).*temp_V)));
                end

            end
            

            count = count - 1;
            if count <= 0
                W = W * (1-skip/(idx+t0));
                V = V * (1-skip/(idx+t0));
                count = skip;
            end
    
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
                fprintf('%d epoch(validation)---processing %dth sample\n',t, j);
             end


            X = test_X(j,:);
            y = -ones(1, class_num);
            y(test_Y(j,:)) = 1;
            
            nz_idx = find(X);

            % pick anchor points
            [anchor_idx, weight] = knn(anchors, X, nearest_neighbor, beta);
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
            mse_llfm_test = mse_llfm_test - sum(log(err));

            [~, label] = max(y_predict);
        
            % accuracy
            if label == test_Y(j,:)
                correct_num = correct_num + 1;
            end
        end


        accuracy_llfm(i,t) = correct_num/num_sample_test;

        rmse_llfm_test(i, t) = (mse_llfm_test / num_sample_test);

        toc;
        fprintf('%d iter(%d epoch)---loss: %f\t accuracy: %f\n', i,t,rmse_llfm_test(i, t),accuracy_llfm(i,t));
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
