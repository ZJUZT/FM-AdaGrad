test_data = load('../UCI/adult_test.csv');
test_X = test_data(:,1:end-1);
test_Y = test_data(:,end);

test_X = test_X';
test_X = mapminmax(test_X,0,1)';
test_Y(test_Y==0)=-1;

%%
train_data = load('../UCI/adult_training.csv');
train_X = train_data(:,1:end-1);
train_Y = train_data(:,end);

train_X = train_X';
train_X = mapminmax(train_X,0,1)';
train_Y(train_Y==0)=-1;