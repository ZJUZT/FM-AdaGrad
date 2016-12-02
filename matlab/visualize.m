[num_sample, p] = size(train_X);

epoch = 1;
plot(mse_fm_sgd(1:epoch*num_sample).^0.5);
hold on
plot(mse_llfm_sgd(1:epoch*num_sample).^0.5);
hold on
% plot(mse_da_llfm_sgd(1:epoch*num_sample).^0.5);
% hold on
plot(mse_dkllfm_sgd(1:epoch*num_sample).^0.5);
xlabel('Number of samples seen');
ylabel('RMSE');
legend('fm','llfm', 'dkllfm');
title('SGD (1 pass)')
grid on;

%%
% [num_sample, p] = size(train_X);
plot(rmse_fm_test(1:epoch));
hold on
plot(rmse_llfm_test(1:epoch));
hold on
plot(rmse_dallfm_test(1:epoch));
xlabel('SGD Epoch');
ylabel('RMSE');
legend('fm','llfm','da\_llfm');
title('Test RMSE')
grid on;
