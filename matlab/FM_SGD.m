rng('default');

% load training data
% train_X, train_Y
% load('training_data_1m');
% load('test_data_1m');
recommendation = 0;
regression = 1;
classification = 2;
task = classification;

if task == recommendation
    [num_sample, ~] = size(train_X);
    p = max(train_X(:,2));
else
    [num_sample, p] = size(train_X);
end

% rand('state',1); 
% randn('state',1);
% 
y_max = max(train_Y);
y_min = min(train_Y);

% parameters
iter_num = 5;
% ml 100k
% learning_rate = 2e3;
% t0 = 1e4;
% skip = 1e3;

% ijcnn
% learning_rate = 1e4;
% t0 = 1e5;
% skip = 1e3;

% magic04
% learning_rate = 1e4;
% t0 = 1e5;
% skip = 1e3;

% covtype
learning_rate = 1e4;
t0 = 1e5;
skip = 1e3;

% ml 100k
% learning_rate = 5e3;
% t0 = 1e5;
% skip = 1e1;

% learning_rate = 1e-2;
% reg = 1e-4;

% banana
% learning_rate = 1e2;
% t0 = 1e5;
% skip = 1e3;

% netflix
% learning_rate = 6e3;
% t0 = 1e5;
% skip = 1e3;

% magic04
% learning_rate = 2e3;
% t0 = 1e5;
% skip = 1e3;

count = skip;

factors_num = 10;

% reg_w = 0;
% reg_v = 0;


  
epoch = 15;

% accelerate the learning process
% momentum = 0.9;


rmse_fm_test = zeros(iter_num, epoch);
rmse_fm_train = zeros(iter_num, epoch);
accuracy_fm = zeros(iter_num, epoch);

% given_sample = 1e4;

% w0_ = 0;
% W_ = 0;
% V_ = 0; 

% T = 1e5;

for i=1:iter_num
    
    tic;
    
    % do shuffle
    
    
    w0 = 0;
    W = zeros(1,p);
    V = 0.1*randn(p,factors_num);
%     
%     w0_ = 0;
%     W_ = zeros(1,p);
%     V_ = zeros(p,factors_num);
    
    mse_fm_sgd = zeros(1,num_sample);
    loss = zeros(1,epoch*num_sample);
    
    % SGD
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

            if task==recommendation
                feature_idx = X_train(j,:);
                X = zeros(1, p);
                X(feature_idx) = 1;
                y = Y_train(j,:);
%                 factor_part = 0 ;
                factor_part = sum(V(feature_idx(1),:).*V(feature_idx(2),:));
                y_predict = w0 + sum(W(feature_idx)) + factor_part;
%                 y_predict = factor_part;
            else
                
                X = X_train(j,:);
                y = Y_train(j,:);
                
                nz_idx = find(X);

                tmp = sum(repmat(X(nz_idx)',1,factors_num).*V(nz_idx,:));
                factor_part = (sum(tmp.^2) - sum(sum(repmat((X(nz_idx)').^2,1,factors_num).*(V(nz_idx,:).^2))))/2;
                y_predict = w0 + W(nz_idx)*X(nz_idx)' + factor_part;

%                 tmp = sum(repmat(X',1,factors_num).*V);
%                 factor_part = (sum(tmp.^2) - sum(sum(repmat((X').^2,1,factors_num).*(V.^2))))/2;
%                 y_predict = w0 + W*X' + factor_part;
            end
            
%             y_predict = min(y_predict, y_max);
%             y_predict = max(y_predict, y_min);

            if task == classification
                err = sigmf(y*y_predict,[1,0]);
            else
                err = y_predict - y;
            end
            

            idx = (t-1)*num_sample + j;

%             idx = j;
            if idx==1
                if task == classification
                    mse_fm_sgd(idx) = -log(err);
                else
                    mse_fm_sgd(idx) = err^2;
                end
                
            else
                if task == classification
                    mse_fm_sgd(idx) = (mse_fm_sgd(idx-1) * (idx - 1) -log(err))/idx;
                else
                    mse_fm_sgd(idx) = (mse_fm_sgd(idx-1) * (idx - 1) + err^2)/idx;
                end
            end

            if task == classification
                rmse_fm_train(i, t) = mse_fm_sgd(idx);
            else
                rmse_fm_train(i,t) = mse_fm_sgd(idx)^0.5;
            end


            % update parameters
%             w0_ = momentum*w0_ + learning_rate * (2 * err);

            if task == recommendation
                w0_ = learning_rate / (idx + t0) * (2* err);
%                 w0_ = learning_rate * (2* err + 2 * reg * w0);
                w0 = w0 - w0_;

%                 W_(feature_idx) = momentum*W_(feature_idx) + learning_rate * (2*err + 2*reg_w*W(feature_idx));
%                 W(feature_idx) = W(feature_idx) - W_(feature_idx);

                W_ = learning_rate / (idx + t0) * (2*err);
%                 W_ = learning_rate * (2*err + 2 * reg * W(feature_idx));
                W(feature_idx) = W(feature_idx) - W_;

%                 V_(feature_idx,:) = momentum*V_(feature_idx,:) + learning_rate * (2*err*((repmat(sum(V(feature_idx,:)),2,1)-V(feature_idx,:))) + 2*reg_v*V(feature_idx,:));
%                 V(feature_idx,:) = V(feature_idx,:) - V_(feature_idx, :);
                V_ = learning_rate / (idx + t0) * (2*err*((repmat(sum(V(feature_idx,:)),2,1)-V(feature_idx,:))));
%                 V_ = learning_rate* (2*err*((repmat(sum(V(feature_idx,:)),2,1)-V(feature_idx,:))) + 2 * reg * V(feature_idx,:));
                V(feature_idx,:) = V(feature_idx,:) - V_;
            end

            if task == classification
                w0_ = learning_rate / (idx + t0) * ((err-1)*y);
%                 w0_ = learning_rate * ((err-1)*y + 2 * reg * w0); 
                w0 = w0 - w0_;
                W_ = learning_rate / (idx + t0) * ((err-1)*y*X(nz_idx));
                W(nz_idx) = W(nz_idx) - W_;
                V_ = learning_rate / (idx + t0) * ((err-1)*y*(repmat(X(nz_idx)',1,factors_num).*(repmat(X(nz_idx)*V(nz_idx,:),length(nz_idx),1)-repmat(X(nz_idx)',1,factors_num).*V(nz_idx,:))));
                V(nz_idx,:) = V(nz_idx,:) - V_;
%                 W_ = learning_rate / (idx + t0) * ((err-1)*y*X);
%                 W_ = learning_rate * ((err-1)*y*X);
%                 W = W - W_;
%                 V_ = learning_rate / (idx + t0) * ((err-1)*y*(repmat(X',1,factors_num).*(repmat(X*V,p,1)-repmat(X',1,factors_num).*V)));
%                 V_ = learning_rate * ((err-1)*y*(repmat(X',1,factors_num).*(repmat(X*V,p,1)-repmat(X',1,factors_num).*V)));
%                 V = V - V_;
            end
            
            if task == regression
                w0_ = learning_rate / (idx + t0) * 2 * err;
                w0 = w0 - w0_;
                W_ = learning_rate / (idx + t0) * (2 * err * X);
                W = W - W_;
                V_ = learning_rate / (idx + t0) * (2*err*(repmat(X',1,factors_num).*(repmat(X*V,p,1)-repmat(X',1,factors_num).*V)));
                V = V - V_;
            end
            
            count = count - 1;
            if count <= 0
                W = W * (1-skip/(idx+t0));
                V = V * (1-skip/(idx+t0));
                count = skip;
            end
        end
    
    % validate
    tic;
    fprintf('validating\n');
    mse = 0.0;
    correct_num = 0;
    [num_sample_test, ~] = size(test_X);
    for k=1:num_sample_test
%         X = test_X(k,:);
%         y = test_Y(k,:);
        if mod(k,1e3)==0
            toc;
            tic;
            fprintf('%d epoch(validation)---processing %dth sample\n',i, k);
        end
 
        if task==recommendation
            X = zeros(1, p);
            feature_idx = test_X(k,:);
            X(feature_idx) = 1;
            y = test_Y(k,:);
%             factor_part = 0;
            % simplify just for recommendation question
            factor_part = sum(V(feature_idx(1),:).*V(feature_idx(2),:));
            y_predict = w0 + sum(W(feature_idx)) + factor_part;
        else 
            X = test_X(k,:);
            y = test_Y(k,:);
            nz_idx = find(X);
            
            tmp = sum(repmat(X(nz_idx)',1,factors_num).*V(nz_idx,:)) ;
%             factor_part = 0;
            factor_part = (sum(tmp.^2) - sum(sum(repmat((X(nz_idx)').^2,1,factors_num).*(V(nz_idx,:).^2))))/2;
            y_predict = w0 + W(nz_idx)*X(nz_idx)' + factor_part;
        end

%         y_predict = min(y_predict, y_max);
%         y_predict = max(y_predict, y_min);
        
        if task == classification
            err = sigmf(y*y_predict,[1,0]);
            mse = mse - log(err);
        else
            err = y_predict - y;
            mse = mse + err.^2;
        end

        if task == classification
            if (y_predict>=0 && y==1) || (y_predict<0&&y==-1)
                correct_num = correct_num + 1;
            end
        end
    end

    if task == classification
        rmse_fm_test(i,t) = (mse / num_sample_test);
    else
        rmse_fm_test(i,t) = (mse / num_sample_test)^0.5;
    end
    
    if task == classification
        accuracy_fm(i,t) = correct_num/num_sample_test;
    end
    
    fprintf('validation done\n');
    toc;
    fprintf('%d iter(%d epoch)---loss: %f\n', i,t,rmse_fm_test(i, t));
    end
end





%%
% plot
plot(mse_fm_sgd,'DisplayName','FM');
legend('-DynamicLegend');
xlabel('Number of samples seen');
ylabel('RMSE');
grid on; 
hold on;  

%%
plot(rmse_fm_test(1,:) ,'k--o','DisplayName','FM');
legend('-DynamicLegend');
% title('Learning Curve on Test Dataset')
hold on;
% plot(rmse_fm_test,'DisplayName','FM\_Test');  
% legend('-DynamicLegend');
xlabel('epoch');
ylabel('logloss');
% legend('FM_Train','FM_Test');
% title('FM\_SGD');
grid on;