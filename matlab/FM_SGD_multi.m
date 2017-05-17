rng('default');

class_num = max(train_Y);
[num_sample, p] = size(train_X);

y_max = max(train_Y);
y_min = min(train_Y);

% parameters
iter_num = 1;

% ml 100k
learning_rate = 1e4;
t0 = 1e5;
skip = 1e1;


count = skip;

factors_num = 10;

epoch = 10;

rmse_fm_test = zeros(iter_num, epoch);
rmse_fm_train = zeros(iter_num, epoch);
accuracy_fm = zeros(iter_num, epoch);

for i=1:iter_num
    
    % SGD
    re_idx = randperm(num_sample);
    X_train = train_X(re_idx,:);
    Y_train = train_Y(re_idx);
    
    tic;
    
    w0 = zeros(class_num, 1);
    W = zeros(class_num, p);
    V = 0.1*randn(class_num, p, factors_num);
    
    mse_fm_sgd = zeros(1,num_sample);
    loss = zeros(1,epoch*num_sample);

    for t=1:epoch
        
        for j=1:num_sample

            if mod(j,1e3)==0
                toc;
                tic;
                fprintf('%d iter(%d epoch)---processing %dth sample\n', i, t, j);
            end
                
            X = X_train(j,:);
            y = -ones(1, class_num);
            y(Y_train(j,:)) = 1;

            nz_idx = find(X);
            
            y_predict = zeros(1, class_num);
            for u = 1:class_num
                tmp = sum(repmat(X(nz_idx)',1,factors_num).*squeeze(V(u,nz_idx,:)));
                factor_part = (sum(tmp.^2) - sum(sum(repmat((X(nz_idx)').^2,1,factors_num).*(squeeze(V(u,nz_idx,:)).^2))))/2;
                y_predict(u) = w0(u) + W(u,nz_idx)*X(nz_idx)' + factor_part;
            end
            

            err = sigmf(y.*y_predict,[1,0]);

            idx = (t-1)*num_sample + j;

            if idx==1
                mse_fm_sgd(idx) = sum(-log(err));
            else
                mse_fm_sgd(idx) = (mse_fm_sgd(idx-1) * (idx - 1) -sum(log(err)))/idx;

            end

            rmse_fm_train(i, t) = mse_fm_sgd(idx);

            % update parameters

            w0_ = learning_rate / (idx + t0) * ((err-1).*y); 
            w0 = w0 - w0_;
            
            W_ = learning_rate / (idx + t0) * (((err-1).*y)'*X(nz_idx));
            W(:,nz_idx) = W(:,nz_idx) - W_;
            
            for u =1 :class_num
                V_ = learning_rate / (idx + t0)...
                    * ((err(u)-1)*y(u)*(repmat(X(nz_idx)',1,factors_num).*(repmat(X(nz_idx)*squeeze(V(u,nz_idx,:)),length(nz_idx),1)-repmat(X(nz_idx)',1,factors_num).*squeeze(V(u,nz_idx,:)))));
                V(u,nz_idx,:) = squeeze(V(u,nz_idx,:)) - V_;
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

        if mod(k,1e4)==0
            toc;
            tic;
            fprintf('%d epoch(validation)---processing %dth sample\n',t, k);
        end
 

        X = test_X(k,:);
        y = -ones(1, class_num);
        y(test_Y(k,:)) = 1;
        
        nz_idx = find(X);

        y_predict = zeros(1, class_num);
        for u = 1:class_num
            tmp = sum(repmat(X(nz_idx)',1,factors_num).*squeeze(V(u,nz_idx,:)));
            factor_part = (sum(tmp.^2) - sum(sum(repmat((X(nz_idx)').^2,1,factors_num).*(squeeze(V(u,nz_idx,:)).^2))))/2;
            y_predict(u) = w0(u) + W(u,nz_idx)*X(nz_idx)' + factor_part;
        end

        err = sigmf(y.*y_predict,[1,0]);
        mse = mse - sum(log(err));

        [~, label] = max(y_predict);
        
        % accuracy
        if label == test_Y(k,:)
            correct_num = correct_num + 1;
        end

    end

    rmse_fm_test(i,t) = (mse / num_sample_test);
    accuracy_fm(i,t) = correct_num/num_sample_test;

    fprintf('validation done\n');
    toc;
    fprintf('%d iter(%d epoch)---loss: %f\t accuracy: %f\n', i,t,rmse_fm_test(i, t),accuracy_fm(i,t));
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
plot(rmse_fm_test ,'k--o','DisplayName','FM');
legend('-DynamicLegend');
% title('Learning Curve on Test Dataset')
hold on;
% plot(rmse_fm_test,'DisplayName','FM\_Test');  
% legend('-DynamicLegend');
xlabel('epoch');
ylabel('RMSE');
% legend('FM_Train','FM_Test');
% title('FM\_SGD');
grid on;