% load training data
% train_X, train_Y
load('training_data');
load('test_data');
[num_sample, p] = size(train_X);
y_max = max(train_Y);
y_min = min(train_Y);

% parameters
iter_num = 10;
learning_rate = 0.01;
factors_num = 10;
reg_w = 0.001;
reg_v = 0.001;

momentum = 0;

w0 = rand();
W = rand(1,p);
V = rand(p,factors_num);

mse_fm_sgd = zeros(1,iter_num*num_sample);

loss = zeros(1,iter_num*num_sample);
rmse_fm_test = zeros(1,iter_num);

w0_ = 0;
W_ = 0;
V_ = 0;

for i=1:iter_num
    
    tic;
    
    % do shuffle
    re_idx = randperm(num_sample);
    X_train = train_X(re_idx,:);
    Y_train = train_Y(re_idx);
    
    % SGD
    
    for j=1:num_sample
        if mod(j,10000)==0
            fprintf('%d epoch---processing %dth sample\n', i, j);
        end
        
        X = X_train(j,:);
        y = Y_train(j,:);
        
        tmp = sum(repmat(X',1,factors_num).*V);
        factor_part = (sum(tmp.^2) - sum(sum(repmat((X').^2,1,factors_num).*(V.^2))))/2;
        y_predict = w0 + W*X' + factor_part;
        
        % prune
%         if y_predict < y_min
%             y_predict = y_min;
%         end
%         
%         if y_predict > y_max
%             y_predict = y_max;
%         end
        
        err = y_predict - y;
        
        idx = (i-1)*num_sample + j;
        loss(idx) = err^2;
        mse_fm_sgd(idx) = sum(loss)/idx;
        
        % update parameters
        w0_ = learning_rate * (2 * err + 2*reg_w*w0);
        w0 = w0 - w0_;
        W_ = learning_rate * (2*err*X + 2*reg_w*W);
        W = W - W_;
        V_ = + learning_rate * (2*err*(repmat(X',1,factors_num).*(repmat(X*V,p,1)-repmat(X',1,factors_num).*V)) + 2*reg_v*V);
        V = V - V_;
    end
    toc;
    
    % validate
    fprintf('validating\n');
    mse = 0.0;
    [num_sample_test, p] = size(test_X);
    for k=1:num_sample_test
        X = test_X(k,:);
        y = test_Y(k,:);

        tmp = sum(repmat(X',1,factors_num).*V) ;
        factor_part = (sum(tmp.^2) - sum(sum(repmat((X').^2,1,factors_num).*(V.^2))))/2;
        y_predict = w0 + W*X' + factor_part;
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