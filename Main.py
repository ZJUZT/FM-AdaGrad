from sklearn.feature_extraction import DictVectorizer

from DataLoader import load_data
from LLFM_SGD import LLFM_SGD

if __name__ == '__main__':
    (train_data, y_train, train_users, train_items) = load_data("ua.base")
    (test_data, y_test, test_users, test_items) = load_data("ua.test")

    v = DictVectorizer()
    X_train = v.fit_transform(train_data)
    X_test = v.transform(test_data)
    llfm_sgd = LLFM_SGD(iter_num=1,
                        learning_rate=0.1,
                        factors_num=10,
                        reg=0.1,
                        anchor_num=100,
                        neighbor_num=10
                        )
    llfm_sgd.train(X_train, y_train)
