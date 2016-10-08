%% load data
% data souce:
% http://files.grouplens.org/datasets/movielens/ml-100k.zip
% data format = (user_id, movie_id, rating, ts)

trained_data = load('ml-100k/ua.base');
trained_X = trained_data(:,1:2);
trained_y = trained_data(:,3);

test_data = load('ml-100k/ua.test');
test_X = test_data(:,1:2);
test_y = test_data(:,3);

num_user = max(trained_X(:,1));
num_mv = max(trained_X(:,2));

% num_user = max(test_X(:,1));
% num_mv = max(test_X(:,2));

%% PMF
fea_num = 10;
learning_rate = 0.1;

% regularization
lam_U = 0.1;
lam_V = 0.1;

U = 0.1*randn(num_user, fea_num);
V = 0.1*randn(num_mv, fea_num);
predict_matrix = U*V';

%% SGD
iter_num = 100;

