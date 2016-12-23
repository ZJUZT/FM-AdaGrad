% load training data
% train_X, train_Y
% load('training_data_1m');
% load('test_data_1m');
[num_sample, ~] = size(train_X);
p = max(train_X(:,2));
y_max = max(train_Y);
y_min = min(train_Y);

% parameters
iter_num = 1;
learning_rate = 0.01; 
factors_num = 10;
reg_w = 1e-3;
reg_v = 1e-3;

epoch = 1;

% momentum = 0;


rmse_fm_test = zeros(1,iter_num);
rmse_fm_train = zeros(1,iter_num);

% w0_ = 0;
% W_ = 0;
% V_ = 0;

for i=1:iter_num
    
    tic;
    
    % do shuffle
    re_idx = randperm(num_sample);
    X_train = train_X(re_idx,:);
    Y_train = train_Y(re_idx);
    
    w0 = rand();
    W = rand(1,p);
    V = rand(p,factors_num);

    mse_fm_sgd = zeros(1,epoch*num_sample);

    loss = zeros(1,epoch*num_sample);
    
    % SGD
    
    for t=1:epoch
        for j=1:num_sample

            if mod(j,1e3)==0
                toc;
                tic;
                fprintf('%d epoch---processing %dth sample\n', i, j);
            end

            % do pack
    %         X = X_train(j,:);
    %         y = Y_train(j,:);

    %         X = zeros(1, p);
            feature_idx = X_train(j,:);
    %         X(feature_idx) = 1;
            y = Y_train(j,:);


    %         tmp = sum(repmat(X',1,factors_num).*V);
    %         factor_part = (sum(tmp.^2) - sum(sum(repmat((X').^2,1,factors_num).*(V.^2))))/2;
    %         y_predict = w0 + W*X' + factor_part;

            % simplify just for recommendation question
            factor_part = sum(V(feature_idx(1),:).*V(feature_idx(2),:));
            y_predict = w0 + sum(W(feature_idx)) + factor_part;

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
    %         mse_fm_sgd(idx) = sum(loss)/idx;
            idx = (t-1)*num_sample + j;
            if idx==1
                mse_fm_sgd(idx) = err^2;
            else
                mse_fm_sgd(idx) = (mse_fm_sgd(idx-1) * (idx - 1) + err^2)/idx;
            end

            rmse_fm_train(i) = mse_fm_sgd(idx)^0.5;

            % update parameters
            w0_ = learning_rate * (2 * err);
            w0 = w0 - w0_;
    %         W_ = learning_rate * (2*err*X + 2*reg_w*W);
    %         W = W - W_;
    %         V_ = learning_rate * (2*err*(repmat(X',1,factors_num).*(repmat(X*V,p,1)-repmat(X',1,factors_num).*V)) + 2*reg_v*V);
    %         V = V - V_;

            W_ = learning_rate * (2*err + 2*reg_w*W(feature_idx));
            W(feature_idx) = W(feature_idx) - W_;
            V_ = learning_rate * (2*err*((repmat(sum(V(feature_idx,:)),2,1)-V(feature_idx,:))) + 2*reg_v*V(feature_idx,:));
            V(feature_idx,:) = V(feature_idx,:) - V_;

        end
    end
    
    % validate
    fprintf('validating\n');
    mse = 0.0;
    [num_sample_test, ~] = size(test_X);
    for k=1:num_sample_test
%         X = test_X(k,:);
%         y = test_Y(k,:);
        if mod(k,1e5)==0
            toc;
            tic;
            fprintf('%d epoch(validation)---processing %dth sample\n',i, k);
        end
%         X = zeros(1, p);
        feature_idx = test_X(k,:);
%         X(feature_idx) = 1;
        y = test_Y(k,:);

%         tmp = sum(repmat(X',1,factors_num).*V) ;
%         factor_part = (sum(tmp.^2) - sum(sum(repmat((X').^2,1,factors_num).*(V.^2))))/2;
%         y_predict = w0 + W*X' + factor_part;

        % simplify just for recommendation question
        factor_part = sum(V(feature_idx(1),:).*V(feature_idx(2),:));
        y_predict = w0 + sum(W(feature_idx)) + factor_part;
        
        % prune
%         if y_predict < y_min
%             y_predict = y_min;
%         end
%          
%         if y_predict > y_max
%             y_predict = y_max;
%         end
        
        err = y_predict - y;
        mse = mse + err.^2;
    end

    rmse_fm_test(i) = (mse / num_sample_test).^0.5;
    fprintf('validation done\n');
end





%%
% plot
plot(mse_fm_sgd.^0.5);
xlabel('Number of samples seen');
ylabel('MSE');
grid on;