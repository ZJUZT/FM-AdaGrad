% load data
% training data
train_raw_data = load('../ml-100k/ua.base');

% calculate total feature
num_user = max(train_raw_data(:,1));
num_movie = max(train_raw_data(:,2));
num_fea = num_user + num_movie;

% get training sampe
% num_sample * num_fea
[num_sample,~] = size(train_raw_data);
train_X = zeros(num_sample,num_fea);

train_raw_data(:,2) = train_raw_data(:,2) + num_user;
for i=1:num_sample
    train_X(i,train_raw_data(i,1:2)) = 1;
end

train_Y = train_raw_data(:,3);
%%
% test data
test_raw_data = load('../ml-100k/ua.test');

% get training sampe
% num_sample * num_fea
[num_sample,~] = size(test_raw_data);
test_X = zeros(num_sample,num_fea);

test_raw_data(:,2) = test_raw_data(:,2) + num_user;
for i=1:num_sample
    test_X(i,test_raw_data(i,1:2)) = 1;
end

test_Y = test_raw_data(:,3);
