%% Global setting
recommendation = 0;
regression = 1;
classification = 2;

adaptive_none = 0;
adaptive_anchor = 1;
adaptive_knn = 2;

factors_num = 10;
iter = 1;
epoch = 1;
task = recommendation;

%% Standard Factorization Machines

% hyper parameters
learning_rate = 1e-2;
reg = 1;

[training_loss_fm, test_loss_fm] = FM(train_X, train_Y, test_X, test_Y,...
    'learning_rate', learning_rate, 'reg', reg, 'factors_num', factors_num, 'iter', iter, 'epoch', epoch, 'task', task);

%% plot
plot(training_loss_fm,'DisplayName','FM\_Train');
legend('-DynamicLegend');
xlabel('Number of samples seen');
ylabel('RMSE');
grid on;
hold on;

%% Locally Linear Factorization Machines

% hyper parameters
learning_rate = 1e-1;
reg = 1;
anchors_num = 10;
nearest_neighbor = 5;
beta = 1;
algorithm = adaptive_none;

[training_loss_llfm, test_loss_llfm] = AdaLLFM(train_X, train_Y, test_X, test_Y,...
    'learning_rate', learning_rate, 'reg', reg, 'factors_num', factors_num, 'iter', iter, 'epoch', epoch, 'task', task,...
    'anchors_num', anchors_num, 'nearest_neighbor', nearest_neighbor, 'beta', beta, 'algorithm',algorithm);

%% Adaptive Anchor Points LLFM
% hyper parameters
learning_rate = 1e-1;
learning_rate_anchor = 1e-3;
reg = 1;
anchors_num = 10;
nearest_neighbor = 5;
beta = 1;
algorithm = adaptive_anchor;

[training_loss_dallfm, test_loss_dallfm] = AdaLLFM(train_X, train_Y, test_X, test_Y,...
    'learning_rate', learning_rate, 'learning_rate_anchor',learning_rate_anchor,...
    'reg', reg, 'factors_num', factors_num, ...
    'iter', iter, 'epoch', epoch, 'task', task,...
    'anchors_num', anchors_num, 'nearest_neighbor', nearest_neighbor, 'beta', beta, 'algorithm',algorithm);

%% Adaptive KNN LLFM
% hyper parameters
learning_rate = 1e-1;
learning_rate_anchor = 1e-3;
reg = 1e1;
anchors_num = 10;
% nearest_neighbor = 5;
LC = 0.5;
algorithm = adaptive_knn;

[training_loss_dkllfm, test_loss_dkllfm, nn] = AdaLLFM(train_X, train_Y, test_X, test_Y,...
    'learning_rate', learning_rate, 'learning_rate_anchor',learning_rate_anchor,...
    'reg', reg, 'factors_num', factors_num, ...
    'iter', iter, 'epoch', epoch, 'task', task,...
    'anchors_num', anchors_num, 'beta', LC, 'algorithm',algorithm);

%%
nn_cum = zeros(1,length(nn));
tmp = 0;
for i=1:length(nn)
    tmp = tmp + nn(i);
    nn_cum(i) = tmp/i;
end
