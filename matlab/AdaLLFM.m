function [training_loss, test_loss, varargout] = AdaLLFM(train_X, train_Y, test_X, test_Y, varargin)
% AdaLLFM: Adaptive Locally linear Factorization Machines
%
%	training_loss : cumulative training loss w.r.t. number of samples seen, log loss for classfication, rmse otherwise
%	test_loss : test loss w.r.t. number of epoch
%	train_X : training data, for 'recommendation' task, just user_id and item_id needed
%	train_Y : label
%	test_X : test data, same with train_X in 'recommendation' task
%
%	[ ... ] = ADALLFM(..., 'PARAM1',val1, 'PARAM2',val2, ...) specifies
%   optional parameter name/value pairs to control the iterative algorithm
%   used by AdaLLFM.  Parameters are:
%
%	'iter' - iteration times, for measure average performance, default: 1
%	'epoch' - epoch, sgd pass in one iteration, default: 1
%	'task' - {'recommendation, 'classification', 'regression'}, default: ''recommendation''
%	'factors_num' - default: 10
%	'reg' - regularization, default: 1e-3
%	'learning_rate' - default: 1e-1
%	'learning_rate_anchor' - default:1e-3
%	'algorithm' - {none, 'anchor_points', 'knn'}
%	'anchors_num' - number of anchor points, defalut: 100
%	'nearest_neighbor' - NN, default: 10, not needed for adaptive knn
%	'LC'/'beta' - default: 1


if nargin < 4
    error('AdaLLFM:TooFewInputs','At least four input arguments required.');
end

rand('state',0); 
randn('state',0);

recommendation = 0;
regression = 1;
classification = 2;

adaptive_none = 0;
adaptive_anchor = 1;
adaptive_knn = 2;

pnames = {'iter', 'epoch', 'task',...
    'learning_rate','learning_rate_anchor', 'reg', 'factors_num',...
    'algorithm', 'anchors_num','nearest_neighbor',...
    'beta'};
dflts = {1, 1, 0,...
    1e-1, 1e-3, 1e-2, 10,...
    0, 10, 3,...
    1};
[~,~,iter,epoch,task,...
    learning_rate,learning_rate_anchor,reg,factors_num,...
    algorithm,anchors_num,nearest_neighbor,...
    beta] = getargs(pnames, dflts, varargin{:});

if task == recommendation
    [num_sample, ~] = size(train_X);
    p = max(train_X(:,2));
else
    [num_sample, p] = size(train_X);
end

training_loss = zeros(iter, epoch);
test_loss = zeros(iter, epoch);
accuracy = zeros(iter, epoch);

% wanna keep record of every sample's number of nn
if algorithm==adaptive_knn
    nn_num = zeros(iter, num_sample * epoch);
end

for i=1:iter
    
    tic;
    
    w0 = zeros(1, anchors_num);
    W = zeros(p,anchors_num);
    V = 0.1*randn(p,factors_num,anchors_num);
    
    % get anchor points
    fprintf('Start K-means...\n');
    if task == recommendation
        [~, anchors, ~, ~, ~] = litekmeans(sparse_matrix(train_X), anchors_num,  'Replicates', 10, 'MaxIter', 1000);
    else
        [~, anchors, ~, ~, ~] = litekmeans(train_X, anchors_num,  'Replicates', 10, 'MaxIter', 1000);
    end
    %     anchors = 0.01* randn(anchors_num, p);
    fprintf('K-means done..\n');
    
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
                fprintf('(algorithm%d)%d iter(%d epoch)---processing %dth sample\n',algorithm, i, t, j);
            end
            
            if task == recommendation
                feature_idx = X_train(j,:);
                X = zeros(1, p);
                X(feature_idx) = 1;
                y = Y_train(j,:);
            else
                X = X_train(j,:);
                y = Y_train(j,:);
            end
            
            
            % pick anchor points
            if algorithm==adaptive_knn
                [anchor_idx, D, lam] = Dynamic_KNN(anchors, X, beta);
                
                nearest_neighbor = length(anchor_idx);
                nn_num(i,(t-1)*num_sample + j) = nearest_neighbor;
                weight = lam - D;
                gamma = weight/sum(weight);
            else
                [anchor_idx, weight] = knn(anchors, X, nearest_neighbor, beta);
                
                gamma = weight/sum(weight);
            end
            
            
            y_anchor = zeros(1, nearest_neighbor);
            if task == recommendation
                for k=1:nearest_neighbor
                    temp_V = squeeze(V(feature_idx,:,anchor_idx(k)));
                    y_anchor(k) = sum(temp_V(1,:).*temp_V(2,:)) + w0(anchor_idx(k)) + sum(W(feature_idx,anchor_idx(k)));
                end
            else
                for k=1:nearest_neighbor
                    temp_V = V(:,:,anchor_idx(k));
                    tmp = sum(repmat(X',1,factors_num).*temp_V);
                    y_anchor(k) = (sum(tmp.^2) - sum(sum(repmat(X'.^2,1,factors_num).*(temp_V.^2))))/2 + w0(anchor_idx(k)) + X*W(:,anchor_idx(k));
                end
            end
            y_predict = gamma * y_anchor';
            
            if task == classification
                err = sigmf(y*y_predict,[1,0]);
            else
                err = y_predict - y;
            end
            
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
                tmp_w0 = w0(anchor_idx);
                w0(anchor_idx) = tmp_w0 - learning_rate * gamma .* 2 * err;
                tmp_W = W(feature_idx,anchor_idx);
                W(feature_idx,anchor_idx) =  tmp_W - learning_rate * repmat(gamma,2,1).*(2*err + 2*reg*tmp_W);
                
                for k=1:nearest_neighbor
                    temp_V = squeeze(V(feature_idx,:,anchor_idx(k)));
                    
                    V(feature_idx,:,anchor_idx(k)) = ...
                        temp_V - learning_rate * gamma(k)* ...
                        (2*err*(repmat(sum(temp_V),2,1)- temp_V) + 2*reg*temp_V);
                end
                
            end
            
            if task == classification
                tmp_w0 = w0(anchor_idx);
                w0(anchor_idx) = tmp_w0 - learning_rate * gamma .* (err-1)*y;
                tmp_W = W(:,anchor_idx);
                W(:,anchor_idx) = tmp_W - learning_rate * repmat(gamma,p,1) .* ((err-1)*y*repmat(X',[1,nearest_neighbor]) + 2*reg*tmp_W);
                for k=1:nearest_neighbor
                    temp_V = squeeze(V(:,:,anchor_idx(k)));
                    V(:,:,anchor_idx(k)) = temp_V - learning_rate * gamma(k) * ((err-1)*y*(repmat(X',1,factors_num).*(repmat(X*temp_V,p,1)-repmat(X',1,factors_num).*temp_V)) + 2*reg*temp_V);
                end
            end
            
            if task == regression
                tmp_w0 = w0(anchor_idx);
                w0(anchor_idx) = tmp_w0 - learning_rate * gamma .* 2 * err;
                tmp_W = W(:,anchor_idx);
                W(:,anchor_idx) = tmp_W - learning_rate * repmat(gamma,p,1) .* (2*err*repmat(X',[1,nearest_neighbor]) + 2*reg*tmp_W);
                for k=1:nearest_neighbor
                    temp_V = squeeze(V(:,:,anchor_idx(k)));
                    V(:,:,anchor_idx(k)) = temp_V - learning_rate * gamma(k) * (2*err*(repmat(X',1,factors_num).*(repmat(X*temp_V,p,1)-repmat(X',1,factors_num).*temp_V)) + 2*reg*temp_V);
                end
            end
            
            % update anchor points
            if algorithm==adaptive_knn
                s = 2 * beta * (repmat(X, nearest_neighbor, 1) - anchors(anchor_idx, :));
                base = -s * sum(weight.*y_anchor);
                base = base + repmat(y_anchor',1,p).* s*sum(weight);
                if task == classification
                    anchors(anchor_idx,:) = anchors(anchor_idx,:) - learning_rate_anchor * (err-1) * y * base/(sum(weight).^2);
                else
                    anchors(anchor_idx,:) = anchors(anchor_idx,:) - learning_rate_anchor * 2 * err * base/(sum(weight).^2);
                end
                
            end
            
            if algorithm==adaptive_anchor
                s = 2 * beta * (repmat(X, nearest_neighbor, 1) - anchors(anchor_idx, :)).*repmat(weight, p, 1)';
                base = -s * sum(weight.*y_anchor);
                base = base + repmat(y_anchor',1,p).* s*sum(weight);
                if task == classification
                    anchors(anchor_idx,:) = anchors(anchor_idx,:) - learning_rate_anchor * ((err-1) * y * base/(sum(weight).^2));
                else
                    anchors(anchor_idx,:) = anchors(anchor_idx,:) - learning_rate_anchor * (2*err * base/(sum(weight).^2));
                end
            end

            training_loss(i,t) = loss(end);
        end
        
        % validate
        fprintf('validating\n');
        
        mse = 0.0;
        correct_num = 0;
        [num_sample_test, ~] = size(test_X);
        
        for j=1:num_sample_test
            if mod(j,1e3)==0
                toc;
                tic;
                fprintf('%d epoch(validation)---processing %dth sample\n',i, j);
            end
            
            if task == recommendation
                X = zeros(1, p);
                feature_idx = test_X(j,:);
                X(feature_idx) = 1;
                y = test_Y(j,:);
            else
                X = test_X(j,:);
                y = test_Y(j,:);
            end
            % pick anchor points
            
            if algorithm==adaptive_knn
                [anchor_idx, D, lam] = Dynamic_KNN(anchors, X, beta);
                nearest_neighbor = length(anchor_idx);
                
                weight = lam - D;
                gamma = weight/sum(weight);
            else
                [anchor_idx, weight] = knn(anchors, X, nearest_neighbor, beta);
                gamma = weight/sum(weight);
            end
            
            y_anchor = zeros(1, nearest_neighbor);
            if task == recommendation
                for k=1:nearest_neighbor
                    temp_V = squeeze(V(feature_idx,:,anchor_idx(k)));
                    y_anchor(k) = sum(temp_V(1,:).*temp_V(2,:)) + w0(anchor_idx(k)) + sum(W(feature_idx,anchor_idx(k)));
                end
            else
                for k=1:nearest_neighbor
                    temp_V = squeeze(V(:,:,anchor_idx(k)));
                    tmp = sum(repmat(X',1,factors_num).*temp_V);
                    y_anchor(k) = (sum(tmp.^2) - sum(sum(repmat(X'.^2,1,factors_num).*(temp_V.^2))))/2 + w0(anchor_idx(k)) + X*W(:,anchor_idx(k));
                end
            end
            
            y_predict = gamma * y_anchor';
            
            if task == classification
                err = sigmf(y*y_predict,[1,0]);
                mse = mse - log(err);
            else
                err = y_predict - y;
                mse = mse + err.^2;
            end

            if task == classification
                if (y_predict>=0 && y==1) || (y_predict<0 && y==-1)
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

if algorithm==adaptive_knn && task == classification
    varargout{1} = accuracy;
    varargout{2} = nn_num;
elseif algorithm==adaptive_knn
    varargout{1} = nn_num;
elseif task==classification
    varargout{1} = accuracy;
else
    
end