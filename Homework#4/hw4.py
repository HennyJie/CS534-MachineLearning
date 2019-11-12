'''
@Author: your name
@Date: 2019-11-09 19:31:31
@LastEditTime: 2019-11-12 09:18:50
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /Machine Learning/Homework#4/hw4.py
'''
import numpy as np
import math
from sklearn.datasets import load_breast_cancer
from sklearn import metrics
from sklearn.model_selection import train_test_split


class GreedyKNN:
    def euclidean_distance(self, instance1, instance2):
        distance = 0
        for i in range(len(instance1)):
            distance += (instance1[i] - instance2[i]) ** 2
        return math.sqrt(distance)

    def kNNpredict(self, X, y, k=5):
        distances = np.zeros([X.shape[0], X.shape[0]])
        y_hat = np.zeros(X.shape[0])

        for i in range(X.shape[0]):
            for j in range(X.shape[0]):
                if i == j:
                    distances[i][j] = float('inf')
                else:
                    distances[i][j] = self.euclidean_distance(X[i, :], X[j, :])

        # sort in each column
        sorted_index = np.argsort(distances, axis=0)

        neighbors = np.zeros([k, X.shape[0]])
        for i in range(k):
            for j in range(X.shape[0]):
                neighbors[i][j] = y[sorted_index[i][j]]

        for i in range(X.shape[0]):
            y_hat[i] = np.sum(neighbors[:, i]) / len(neighbors[:, i])

        return y_hat

    def get_feature_order(self, X, y, k=5):

        n, m = X.shape
        feature_lst = []

        while len(feature_lst) < m:
            max_auroc = 0.0
            max_var = -1
            for j in range(m):
                if j in feature_lst:
                    continue
                # Implement your own kNNpredict function
                # The function should return kNN prediction for the given X, y, and k
                y_hat = self.kNNpredict(X[:, feature_lst + [j]], y, k)  # TODO

                # Measure AUROC with y_hat and y
                auroc = metrics.roc_auc_score(y, y_hat)
                if auroc > max_auroc:
                    max_auroc = auroc
                    max_var = j
            print("max_var: ", max_var)

            feature_lst.append(max_var)

        print("feature list: ", feature_lst)

        return feature_lst


data_breast_cancer = load_breast_cancer()
X = data_breast_cancer.data

# X_mu = X.mean(axis=0)
# X_sigma = X.std(axis=0)
# X = (X-X_mu)/X_sigma

y = data_breast_cancer.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=100)
print("X_train shape: ", X_train.shape)
print("y_train shape: ", y_train.shape)
knn = GreedyKNN()
k = 5
feature_lst = knn.get_feature_order(X_train, y_train, k)

y_pred = knn.kNNpredict(X_test[:, feature_lst], y_test, k)
AUC = metrics.roc_auc_score(y_test, y_pred)
print("AUC: ", AUC)
