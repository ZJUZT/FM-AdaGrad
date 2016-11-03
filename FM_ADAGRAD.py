# -*- coding: utf-8 -*-
import numpy as np
import math


class FM_ADAGRAD:
    def __init__(self,
                 iter_num,
                 learning_rate,
                 factors_num,
                 reg,
                 verbose=True):

        # 迭代次数
        self.iter_num = iter_num

        # 学习速率
        self.learning_rate = learning_rate

        # 分解器feature个数
        self.factors_num = factors_num

        # lambda
        self.reg = reg

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
        self.w0 = sum(np.random.rand(1))  # bias

        # feature bias
        self.W = np.random.rand(1, p)

        # feature
        self.V = np.random.rand(p, self.factors_num)

        self.y_max = np.max(y_)
        self.y_min = np.min(y_)

        # keep track of all parameters
        w0_gradients = 0.0
        W_gradients = np.zeros((1, p))
        V_gradients = np.zeros((p, self.factors_num))

        # avoid 0 numerator
        epison = 1e-8

        for j in xrange(self.iter_num):

            loss_sgd = []

            # shuffle
            reidx = np.random.permutation(n)
            X_train = X_[reidx, :]
            y_train = y_[reidx]

            for i in xrange(n):

                if self.verbose and i % 1000 == 0:
                    print 'processing ' + str(i) + 'th sample...'

                X = X_train[i, :]
                y = y_train[i]

                # too slow
                #     y_predict = (w0 + W*X.T + ((X.T*X).multiply((np.triu(V.dot(V.T),1)))).sum().sum())[0,0]

                tmp = np.sum(X.T.multiply(self.V), axis=0)
                factor_part = (np.sum(np.multiply(tmp, tmp)) - np.sum(
                    (X.T.multiply(X.T)).multiply(np.multiply(self.V, self.V)))) / 2
                y_predict = self.w0 + np.sum(self.W * X.T) + factor_part

                #                 print y_predict

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
                gradient = 2 * diff * (1)
                w0_gradients += np.square(gradient)
                self.w0 -= self.learning_rate / np.sqrt(w0_gradients + epison) * gradient

                # update W
                gradient = 2 * diff * (X)
                W_gradients += gradient.multiply(gradient)
                self.W = self.W - gradient.multiply(self.learning_rate / np.sqrt(W_gradients + epison))

                #                 print W_gradients.shape
                #                 print gradient.shape

                # update V
                gradient = 2 * diff * (X.T.multiply((np.tile(X * self.V, (p, 1)) - X.T.multiply(self.V))))
                V_gradients += np.multiply(gradient, gradient)
                self.V = self.V - np.multiply(self.learning_rate / np.sqrt(V_gradients + epison), gradient)

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
            y_predict = self.w0 + np.sum(self.W * X.T) + factor_part

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
