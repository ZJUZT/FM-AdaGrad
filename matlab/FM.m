function [training_loss, test_loss, varargout] = FM(train_X, train_Y, test_X, test_Y, varargin)
% FM: Standard Factorization Machines
%
%	training_loss : cumulative training loss w.r.t. number of samples seen, log loss for classfication, rmse otherwise
%	test_loss : test loss w.r.t. number of epoch
%	train_X : training data, for 'recommendation' task, just user_id and item_id needed
%	train_Y : label
%	test_X : test data, same with train_X in 'recommendation' task
%
%	[ ... ] = FM(..., 'PARAM1',val1, 'PARAM2',val2, ...) specifies
%   optional parameter name/value pairs to control the iterative algorithm
%   used by AdaLLFM.  Parameters are:
%
%	'iter' - iteration times, for measure average performance, default: 1
%	'epoch' - epoch, sgd pass in one iteration, default: 1
%	'task' - {'recommendation, 'classification', 'regression'}, default: ''recommendation''
%	'factors_num' - default: 10
%	'reg' - regularization, default: 1e-3
%	'learning_rate' - default: 1e-2


if nargin < 4
    error('FM:TooFewInputs','At least four input arguments required.');
end

rand('state',0); 
randn('state',0);

recommendation = 0;
regression = 1;
classification = 2;

pnames = {'iter', 'epoch', 'task', 'learning_rate', 'reg', 'factors_num'};
dflts = {1, 1, 0, 1e-2, 1e-3, 10};
[~,~,iter,epoch,task,learning_rate,reg,factors_num] = getargs(pnames, dflts, varargin{:});

if task == recommendation
    [num_sample, ~] = size(train_X);
    p = max(train_X(:,2));
else
    [num_sample, p] = size(train_X);
end

training_loss = zeros(iter, epoch);
test_loss = zeros(iter, epoch);
accuracy = zeros(iter, epoch);


for i=1:iter
    
    tic;
    w0 = 0;
    W = zeros(1,p);
    V = 0.1*randn(p,factors_num);
    
    % SGD
    
    for t=1:epoch
        
        % do shuffle
        re_idx = randperm(num_sample);
        X_train = train_X(re_idx,:);
        Y_train = train_Y(re_idx);
        loss = zeros(1, num_sample);
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
                
                factor_part = sum(V(feature_idx(1),:).*V(feature_idx(2),:));
                y_predict = w0 + sum(W(feature_idx)) + factor_part;
            else
                X = X_train(j,:);
                y = Y_train(j,:);
                
                tmp = sum(repmat(X',1,factors_num).*V);
                %                 % factor_part = 0;
                factor_part = (sum(tmp.^2) - sum(sum(repmat((X').^2,1,factors_num).*(V.^2))))/2;
                y_predict = w0 + W*X' + factor_part;
            end
            
            if task == classification
                err = sigmf(y*y_predict,[1,0]);
            else
                err = y_predict - y;
            end
            
            
            % idx = (t-1)*num_sample + j;
            idx = j;
            if idx==1
                if task == classification
                    loss(idx) = -log(err);
                else
                    loss(idx) = abs(err);
                end
                
            else
                if task == classification
                    loss(idx) = (loss(idx-1) * (idx - 1) -log(err))/idx;
                else
                    loss(idx) = ((loss(idx-1)^2 * (idx - 1) + err^2)/idx)^0.5;
                end
            end
            
            % update parameters
            
            if task == recommendation
                w0_ = learning_rate * 2* err;
                w0 = w0 - w0_;
                W_ = learning_rate * (2*err + 2*reg*W(feature_idx));
                W(feature_idx) = W(feature_idx) - W_;
                V_ = learning_rate * (2*err*((repmat(sum(V(feature_idx,:)),2,1)-V(feature_idx,:))) + 2*reg*V(feature_idx,:));
                V(feature_idx,:) = V(feature_idx,:) - V_;
            end
            
            if task == classification
                w0_ = learning_rate * (err-1)*y;
                w0 = w0 - w0_;
                W_ = learning_rate * ((err-1)*y*X + 2*reg*W);
                W = W - W_;
                V_ = learning_rate * ((err-1)*y*(repmat(X',1,factors_num).*(repmat(X*V,p,1)-repmat(X',1,factors_num).*V)) + 2*reg*V);
                V = V - V_;
            end
            
            if task == regression
                w0_ = learning_rate * 2 * err;
                w0 = w0 - w0_;
                W_ = learning_rate * (2 * err * X + 2*reg*W);
                W = W - W_;
                V_ = learning_rate * (2*err*(repmat(X',1,factors_num).*(repmat(X*V,p,1)-repmat(X',1,factors_num).*V)) + 2*reg*V);
                V = V - V_;
            end
        end

        training_loss(i,t) = loss(end);
        
        % validate
        fprintf('validating\n');
        
        mse = 0.0;
        correct_num = 0;
        [num_sample_test, ~] = size(test_X);
        
        for k=1:num_sample_test
            if mod(k,1e5)==0
                toc;
                tic;
                fprintf('%d epoch(validation)---processing %dth sample\n',i, k);
            end
            
            if task==recommendation
%                 X = zeros(1, p);
                feature_idx = test_X(k,:);
%                 X(feature_idx) = 1;
                y = test_Y(k,:);
                
                % simplify just for 'recommendation' question
                factor_part = sum(V(feature_idx(1),:).*V(feature_idx(2),:));
                y_predict = w0 + sum(W(feature_idx)) + factor_part;
            else
                X = test_X(k,:);
                y = test_Y(k,:);
                tmp = sum(repmat(X',1,factors_num).*V);
                factor_part = (sum(tmp.^2) - sum(sum(repmat((X').^2,1,factors_num).*(V.^2))))/2;
                y_predict = w0 + W*X' + factor_part;
            end
            
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
            test_loss(i,t) = (mse / num_sample_test);
        else
            test_loss(i,t) = (mse / num_sample_test)^0.5;
        end

        if task == classification
            accuracy(i,t) = correct_num/num_sample_test;
        end
        
        fprintf('validation done\n');
    end
end

varargout{1} = accuracy;





