# -*- coding: utf-8 -*-
import math

import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans


class LLFM_SGD:
    def __init__(self,
                 iter_num,
                 learning_rate,
                 factors_num,
                 reg_w,
                 reg_v,
                 anchor_num,
                 neighbor_num,
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

        # anchor point 的个数
        self.anchor_num = anchor_num

        # find k-nearest anchor points
        self.neighbor_num = neighbor_num

        # anchor point
        self.anchor_points = np.array([0])

    def knn(self, x):
        dist = np.linalg.norm(self.anchor_points - x, axis=1)
        idx = np.argsort(dist)
        dist = np.exp(-dist[idx[:self.neighbor_num]])
        gamma = dist / np.sum(dist)
        return gamma, idx[:self.neighbor_num]

    def train(self, X_, y_):

        (n, p) = X_.shape

        self.mse = []

        # anchor bias
        self.w0 = np.random.rand(self.anchor_num, 1)

        # feature bias
        #         self.W = np.random.rand(1,p)
        # local coding
        self.W = np.random.rand(self.anchor_num, p)

        # feature
        self.V = np.random.rand(self.anchor_num, p, self.factors_num)

        self.y_max = np.max(y_)
        self.y_min = np.min(y_)

        if self.verbose:
            print 'performing K-means...'

        # K-means get anchor points

        # kmeans = KMeans(n_clusters=self.anchor_num, random_state=0).fit(X_)

        # mini batch
        kmeans = MiniBatchKMeans(n_clusters=self.anchor_num, random_state=0).fit(X_)
        self.anchor_points = kmeans.cluster_centers_

        if self.verbose:
            print 'K-means done...'

        for j in xrange(self.iter_num):

            loss_sgd = []

            # shuffle
            re_idx = np.random.permutation(n)
            x_train = X_[re_idx, :]
            y_train = y_[re_idx]

            for i in xrange(n):

                if self.verbose and i % 1000 == 0:
                    print 'processing ' + str(i) + 'th sample...'

                X = x_train[i, :]
                y = y_train[i]

                (gamma, idx) = self.knn(X)

                X = X.toarray()

                # if self.verbose:
                #     print 'k-nearest neighbors found...'

                # reshape V to 2-dimension

                # V = self.V[idx, :, :].reshape(self.neighbor_num*p, self.factors_num)

                # X_repmat = np.tile(X.toarray(), (self.neighbor_num, 1))

                factor_part = 0.0

                for k in xrange(self.neighbor_num):
                    tmp = np.sum(X.T * self.V[k], axis=0)

                    factor_part += gamma[k] * (np.sum(tmp * tmp) - np.sum(
                        (X.T * X.T) * (self.V[idx[k]] * self.V[idx[k]]))) / 2

                y_predict = np.sum(
                    np.dot(np.array([gamma]), self.w0[idx, :]) + np.dot(np.array([gamma]),
                                                                        np.dot(self.W[idx, :], X.T)) + factor_part)

                # prune
                if y_predict < self.y_min:
                    y_predict = self.y_min

                if y_predict > self.y_max:
                    y_predict = self.y_max

                diff = np.sum(y_predict - y)
                loss_sgd.append(math.pow(diff, 2))

                # update mse
                self.mse.append(sum(loss_sgd) / len(loss_sgd))

                # update w0
                self.w0[idx, :] -= np.dot(gamma, self.learning_rate * (2 * diff * 1 + 2 * self.reg_w * self.w0[idx, :]))

                # update W
                self.W[idx, :] -= np.dot(gamma, self.learning_rate * (2 * diff * X + 2 * self.reg_w * self.W[idx, :]))

                # update V
                for k in xrange(self.neighbor_num):
                    self.V[idx[k]] -= gamma[k] * self.learning_rate * (2 * diff * (
                        X.T * (np.dot(X, self.V[idx[k]]) - X.T * self.V[idx[k]])) + 2 * self.reg_v * self.V[idx[k]])

                # self.V -= gamma * self.learning_rate * (2 * diff * (
                #         X.T * (np.dot(X, self.V[idx]) - X.T * self.V[idx])) + 2 * self.reg_v * self.V[idx])

    def validate(self, x_, y_):
        (n, p) = x_.shape

        mse = []
        loss_sgd = []

        for i in xrange(n):

            if self.verbose and i % 1000 == 0:
                print 'processing ' + str(i) + 'th sample...'

            x = x_[i, :]
            y = y_[i]

            tmp = np.sum(x.T.multiply(self.V), axis=0)
            factor_part = (np.sum(np.multiply(tmp, tmp)) - np.sum(
                (x.T.multiply(x.T)).multiply(np.multiply(self.V, self.V)))) / 2
            y_predict = self.w0 + np.sum(self.W * x.T) + factor_part

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
