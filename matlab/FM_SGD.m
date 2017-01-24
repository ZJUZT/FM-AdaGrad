% load training data
% train_X, train_Y
% load('training_data_1m');
% load('test_data_1m');
[num_sample, p] = size(train_X);
% p = max(train_X(:,2));
% y_max = max(train_Y);
% y_min = min(train_Y);

% parameters
iter_num = 1;
learning_rate = 1e-2;
factors_num = 10;

reg_w = 1e-3;
reg_v = 1e-3;

epoch = 1;

% accelerate the learning process
% momentum = 0.8;


rmse_fm_test = zeros(iter_num, epoch);
rmse_fm_train = zeros(iter_num, epoch);
accuracy_fm = zeros(iter_num, epoch);

given_sample = 1e4;

% w0_ = 0;
% W_ = 0;
% V_ = 0; 

for i=1:iter_num
    
    tic;
    
    % do shuffle
    
    
    w0 = 0.1*randn();
    W = 0.1*randn(1,p);
    V = 0.1*randn(p,factors_num);
    
%     w0_ = 0;
%     W_ = zeros(1,p);
%     V_ = zeros(p,factors_num);
    
    mse_fm_sgd = zeros(1,num_sample);
    loss = zeros(1,epoch*num_sample);
    
    % SGD
    
    for t=1:epoch
        
        re_idx = randperm(num_sample);
        X_train = train_X(re_idx,:);
        Y_train = train_Y(re_idx);
        for j=1:num_sample

            if mod(j,1e3)==0
                toc;
                tic;
                fprintf('%d iter(%d epoch)---processing %dth sample\n', i, t, j);
            end

%             feature_idx = X_train(j,:);
    %         X(feature_idx) = 1;
%             y = Y_train(j,:);

            X = X_train(j,:);
            y = Y_train(j,:);

            tmp = sum(repmat(X',1,factors_num).*V);
            factor_part = (sum(tmp.^2) - sum(sum(repmat((X').^2,1,factors_num).*(V.^2))))/2;
            y_predict = w0 + W*X' + factor_part;

            % simplify just for recommendation question
%             factor_part = sum(V(feature_idx(1),:).*V(feature_idx(2),:));
%             y_predict = w0 + sum(W(feature_idx)) + factor_part;

            err_r = y_predict - y;

            % classification
            err_c = sigmf(y*y_predict,[1,0]);

%             idx = (t-1)*num_sample + j;

            idx = j;
            if idx==1
                % mse_fm_sgd(idx) = err^2;
                mse_fm_sgd(idx) = -log(err_c);
            else
                % mse_fm_sgd(idx) = (mse_fm_sgd(idx-1) * (idx - 1) + err^2)/idx;
                mse_fm_sgd(idx) = (mse_fm_sgd(idx-1) * (idx - 1) -log(err_c))/idx;
            end

            % rmse_fm_train(i, t) = mse_fm_sgd(idx)^0.5;
            rmse_fm_train(i, t) = mse_fm_sgd(idx);

            % update parameters
%             w0_ = momentum*w0_ + learning_rate * (2 * err);
            w0_ = learning_rate * (err_c-1)*y;
            w0 = w0 - w0_;
            W_ = learning_rate * ((err_c-1)*y*X + 2*reg_w*W);
            W = W - W_;
            V_ = learning_rate * ((err_c-1)*y*(repmat(X',1,factors_num).*(repmat(X*V,p,1)-repmat(X',1,factors_num).*V)) + 2*reg_v*V);
            V = V - V_;

% %             W_(feature_idx) = momentum*W_(feature_idx) + learning_rate * (2*err + 2*reg_w*W(feature_idx));
%             W_ = learning_rate * (2*err + 2*reg_w*W(feature_idx));
%             W(feature_idx) = W(feature_idx) - W_;
% %             V_(feature_idx,:) = momentum*V_(feature_idx,:) + learning_rate * (2*err*((repmat(sum(V(feature_idx,:)),2,1)-V(feature_idx,:))) + 2*reg_v*V(feature_idx,:));
%             V_ = learning_rate * (2*err*((repmat(sum(V(feature_idx,:)),2,1)-V(feature_idx,:))) + 2*reg_v*V(feature_idx,:));
%             V(feature_idx,:) = V(feature_idx,:) - V_;
        end
    
    % validate
    fprintf('validating\n');
    mse = 0.0;
    correct_num = 0;
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
%         feature_idx = test_X(k,:);
%         X(feature_idx) = 1;
%         y = test_Y(k,:);
        
        X = test_X(k,:);
        y = test_Y(k,:);

        tmp = sum(repmat(X',1,factors_num).*V) ;
        factor_part = (sum(tmp.^2) - sum(sum(repmat((X').^2,1,factors_num).*(V.^2))))/2;
        y_predict = w0 + W*X' + factor_part;

        % simplify just for recommendation question
%         factor_part = sum(V(feature_idx(1),:).*V(feature_idx(2),:));
%         y_predict = w0 + sum(W(feature_idx)) + factor_part;
        
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

        if (y_predict>=0 && y==1) || (y_predict<0&&y==-1)
            correct_num = correct_num + 1;
        end

        % mse = mse + err.^2;
        mse = mse - log(err_c);
    end

    rmse_fm_test(i,t) = (mse / num_sample_test);
    accuracy_fm(i,t) = correct_num/num_sample_test;
    fprintf('validation done\n');
    end
end





%%
% plot
plot(mse_fm_sgd,'DisplayName','FM\_Train');
legend('-DynamicLegend');
xlabel('Number of samples seen');
ylabel('RMSE');
grid on;
hold on;  

%%
plot(rmse_fm_train,'DisplayName','FM\_Train');
legend('-DynamicLegend');
hold on;
plot(rmse_fm_test,'DisplayName','FM\_Test');
legend('-DynamicLegend');
xlabel('epoch');
ylabel('RMSE');
% legend('FM_Train','FM_Test');
title('FM\_SGD');
grid on;