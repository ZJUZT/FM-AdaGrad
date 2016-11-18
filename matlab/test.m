plot(mse_fm_sgd.^0.5);
hold on
plot(mse_llfm_sgd.^0.5);
hold on
plot(mse_da_llfm_sgd.^0.5);
xlabel('Number of samples seen');
ylabel('RMSE');
legend('fm','llfm','da\_llfm');
title('SGD (1 pass)')
grid on;