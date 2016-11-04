# -*- coding: utf-8 -*-
import numpy as np
import math


class FM_SGD:
    def __init__(self,
                 iter_num,
                 learning_rate,
                 factors_num,
                 reg_w,
                 reg_v,
                 verbose=True):

        # 迭代次数
        self.iter_num = iter_num

        # 学习速率
        self.learning_rate = learning_rate

        # 分解器feature个数
        self.factors_num = factors_num

        # lambda
        self.reg_w = reg_w

        self.reg_v = reg_v

        # 输出执行信息
        self.verbose = verbose

        # global bias
        self.w0 = 0

        # feature bias
        self.W = 0

        # feature
        self.V = 0

        # 训练过程中的mse
        self.mse = []

        # target y的最大值与最小值，for prune
        self.y_max = 0.0
        self.y_min = 0.0

    def train(self, X_, y_):

        (n, p) = X_.shape

        self.mse = []

        # global bias
        self.w0 = np.sum(np.random.rand(1, 1))  # bias

        # feature bias
        self.W = np.random.rand(1, p)

        # feature
        self.V = np.random.rand(p, self.factors_num)

        self.y_max = np.max(y_)
        self.y_min = np.min(y_)

        for j in xrange(self.iter_num):

            loss_sgd = []

            # shuffle
            reidx = np.random.permutation(n)
            X_train = X_[reidx, :]
            y_train = y_[reidx]

            for i in xrange(n):

                if self.verbose and i % 10000 == 0:
                    print 'processing ' + str(i) + 'th sample...'

                X = X_train[i, :]
                y = y_train[i]

                # too slow
                #     y_predict = (w0 + W*X.T + ((X.T*X).multiply((np.triu(V.dot(V.T),1)))).sum().sum())[0,0]

                X = X.toarray()

                tmp = np.sum(X.T * self.V, axis=0)
                factor_part = (np.sum(tmp * tmp) - np.sum(
                    (X.T*X.T)*(self.V * self.V))) / 2
                y_predict = self.w0 + np.sum(np.dot(self.W, X.T)) + factor_part

                # prune
                if y_predict < self.y_min:
                    y_predict = self.y_min

                if y_predict > self.y_max:
                    y_predict = self.y_max

                diff = y_predict - y
                loss_sgd.append(math.pow(diff, 2))

                # update mse
                self.mse.append(sum(loss_sgd) / len(loss_sgd))

                # update w0
                self.w0 -= self.learning_rate * (2 * diff * 1 + 2*self.reg_w*self.w0)

                # update W
                self.W -= self.learning_rate * (2 * diff * X + 2*self.reg_w*self.W)

                # update V
                self.V -= self.learning_rate * (2 * diff * (
                    X.T * (np.dot(X, self.V) - X.T * self.V)) + 2 * self.reg_v * self.V)

    def validate(self, X_, y_):
        (n, p) = X_.shape

        mse = []
        loss_sgd = []

        for i in xrange(n):

            if self.verbose and i % 1000 == 0:
                print 'prossing ' + str(i) + 'th sample...'

            X = X_[i, :]
            y = y_[i]

            # too slow
            #     y_predict = (w0 + W*X.T + ((X.T*X).multiply((np.triu(V.dot(V.T),1)))).sum().sum())[0,0]

            tmp = np.sum(X.T.multiply(self.V), axis=0)
            factor_part = (np.sum(np.multiply(tmp, tmp)) - np.sum(
                (X.T.multiply(X.T)).multiply(np.multiply(self.V, self.V)))) / 2
            y_predict = self.w0 + self.W * X.T + factor_part

            #                 print y_predict

            # prune
            if y_predict < self.y_min:
                y_predict = self.y_min

            if y_predict > self.y_max:
                y_predict = self.y_max

            diff = y_predict - y
            loss_sgd.append(math.pow(diff, 2))

            # update mse
            mse.append(sum(loss_sgd) / len(loss_sgd))
        return mse
