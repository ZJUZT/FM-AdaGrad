% load training data
% train_X, train_Y
load('training_data');
load('test_data');
[num_sample, p] = size(train_X);
y_max = max(train_Y);
y_min = min(train_Y);

% parameters
iter_num = 1;
learning_rate = 0.001;
factors_num = 10;
reg_w = 0;
reg_v = 0;

w0 = rand();
W = rand(1,p);
V = rand(p,factors_num);

rmse = zeros(1,iter_num*num_sample);
rmse_test = zeros(1, iter_num);
loss = zeros(1,iter_num*num_sample);

for i=1:iter_num
    
    tic;
    
    % do shuffle
    re_idx = randperm(num_sample);
    X_train = train_X(re_idx,:);
    Y_train = train_Y(re_idx);
    
    % SGD
    
    for j=1:num_sample
        if mod(j,10000)==0
            fprintf('processing %dth sample\n', j);
        end
        
        X = X_train(j,:);
        y = Y_train(j,:);
        
        tmp = sum(repmat(X',1,factors_num).*V);
        factor_part = (sum(tmp.^2) - sum(sum(repmat((X').^2,1,factors_num).*(V.^2))))/2;
        y_predict = w0 + W*X' + factor_part;
        
        % prune
        if y_predict < y_min
            y_predict = y_min;
        end
        
        if y_predict > y_max
            y_predict = y_max;
        end
        
        err = y_predict - y;
        
        idx = (i-1)*num_sample + j;
        loss(idx) = err^2;
        rmse(idx) = sum(loss)/idx;
        
        % update parameters
        w0 = w0 - learning_rate * (2 * err + 2*reg_w*w0);
        W = W - learning_rate * (2*err*X + 2*reg_w*W);
        V = V - learning_rate * (2*err*(repmat(X',1,factors_num).*(repmat(X*V,p,1)-repmat(X',1,factors_num).*V)) + 2*reg_v*V);
    end
    toc;
end

%%
% plot
plot(rmse);
xlabel('Number of samples seen');
ylabel('MSE');
grid on;