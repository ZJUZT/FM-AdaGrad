% movie length 100k
ml_100k_training = 'data/movielens/training_data_100k';
ml_100k_test = 'data/movielens/test_data_100k';

% movie length 1m
ml_1m_training = 'data/movielens/training_data_1m';
ml_1m_test = 'data/movielens/test_data_1m';

% movie length 10m
ml_10m_training = 'data/movielens/training_data_10m';
ml_10m_test = 'data/movielens/test_data_10m';

% amazon video
amazon_video_train = 'data/amazon/training_data_video';
amazon_video_test = 'dataamazon/test_data_video';

% netflix subset
netflix_train_5K5K = 'data/netflix/training_data5K5K';
netflix_test_5K5K = 'data/netflix/test_data5K5K';

%% classification task
% adult
adult_train = 'data/adult/training_data';
adult_test = 'data/adult/test_data';

%% 
training_data = ml_100k_training;
test_data = ml_100k_test;
% 
% training_data = ml_1m_training;
% test_data = ml_1m_test;

% training_data = amazon_video_train;
% test_data = amazon_video_test;

% training_data = netflix_train_5K5K;
% test_data = netflix_test_5K5K;

% training_data = adult_train;
% test_data = adult_test;

load(training_data);
load(test_data); 