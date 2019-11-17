'''
@Author: Hejie Cui
@Date: 2019-11-09 19:31:31
@LastEditTime: 2019-11-17 16:05:33
@FilePath: /Machine Learning/Homework#4/hw4.py
'''
import numpy as np
import math
from sklearn.datasets import load_breast_cancer
from sklearn import metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class GreedyKNN:
    def kNNpredict(self, X_train, y_train, X_test, k=5, train_status=True):
        if train_status:
            distances = np.zeros([X_train.shape[0], X_train.shape[0]])
            y_hat = np.zeros(X_train.shape[0])

            for i in range(X_train.shape[0]):
                distances[:, i] = np.linalg.norm(X_train - X_train[i], axis=1)
                distances[i][i] = float('inf')

            # sort in each column
            sorted_index = np.argsort(distances, axis=0)
            neighbors = np.zeros([k, X_train.shape[0]])
            for i in range(k):
                for j in range(X_train.shape[0]):
                    neighbors[i][j] = y_train[sorted_index[i][j]]

            for i in range(X_train.shape[0]):
                y_hat[i] = np.mean(neighbors[:, i])
            return y_hat
        else:
            distances = np.zeros([X_train.shape[0], X_test.shape[0]])
            y_hat = np.zeros(X_test.shape[0])
            for i in range(X_test.shape[0]):
                distances[:, i] = np.linalg.norm(X_train - X_test[i], axis=1)

            # sort in each column
            sorted_index = np.argsort(distances, axis=0)
            neighbors = np.zeros([k, X_test.shape[0]])
            for i in range(k):
                for j in range(X_test.shape[0]):
                    neighbors[i][j] = y_train[sorted_index[i][j]]

            for i in range(X_test.shape[0]):
                y_hat[i] = np.mean(neighbors[:, i])
            return y_hat

    def get_feature_order(self, X_train, y_train, X_test, k=5):

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
                y_hat = self.kNNpredict(
                    X_train[:, feature_lst + [j]], y_train, X_test, k, train_status=True)

                # Measure AUROC with y_hat and y
                auroc = metrics.roc_auc_score(y_train, y_hat)
                if auroc > max_auroc:
                    max_auroc = auroc
                    max_var = j
            feature_lst.append(max_var)

        return feature_lst


data_breast_cancer = load_breast_cancer()
X = data_breast_cancer.data
y = data_breast_cancer.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=100)
knn = GreedyKNN()

# change the number of features, t
k = 5
# change t, without features standardized
feature_lst_nonstandard_t = knn.get_feature_order(X_train, y_train, X_test, k)
AUC_nonstandard_change_t = np.zeros(X_train.shape[1])
for t in range(1, len(feature_lst_nonstandard_t)+1):
    y_pred = knn.kNNpredict(
        X_train[:, feature_lst_nonstandard_t[:t]], y_train, X_test[:, feature_lst_nonstandard_t[:t]], k, train_status=False)
    AUC_nonstandard_change_t[t-1] = metrics.roc_auc_score(y_test, y_pred)
print("AUC_nonstandard_change_t: ", AUC_nonstandard_change_t)

# change t, with features standardized
X_train_mu = X_train.mean(axis=0)
X_train_sigma = X_train.std(axis=0)
X_train_standardized = (X_train-X_train_mu)/X_train_sigma
X_test_standardized = (X_test-X_train_mu)/X_train_sigma

feature_lst_standard_t = knn.get_feature_order(
    X_train_standardized, y_train, X_test_standardized, k)
AUC_standard_change_t = np.zeros(X_train.shape[1])
for t in range(1, len(feature_lst_standard_t)+1):
    y_pred = knn.kNNpredict(
        X_train_standardized[:, feature_lst_standard_t[:t]], y_train, X_test_standardized[:, feature_lst_standard_t[:t]], k, train_status=False)
    AUC_standard_change_t[t-1] = metrics.roc_auc_score(y_test, y_pred)
print("AUC_standard_change_t: ", AUC_standard_change_t)

# change the number of neighbors, k
AUC_nonstandard_change_k = np.zeros(10)
AUC_standard_change_k = np.zeros(10)

for k in range(1, 11):
    # change k, without features standardized
    feature_lst_nonstandard_k = knn.get_feature_order(
        X_train, y_train, X_test, k)
    y_pred = knn.kNNpredict(X_train[:, feature_lst_nonstandard_k], y_train,
                            X_test[:, feature_lst_nonstandard_k], k, train_status=False)
    AUC_nonstandard_change_k[k-1] = metrics.roc_auc_score(y_test, y_pred)

    # change k, with features standardized
    feature_lst_standard_k = knn.get_feature_order(
        X_train_standardized, y_train, X_test_standardized, k)
    y_pred = knn.kNNpredict(
        X_train_standardized[:, feature_lst_standard_k], y_train, X_test_standardized[:, feature_lst_standard_k], k, train_status=False)
    AUC_standard_change_k[k-1] = metrics.roc_auc_score(y_test, y_pred)

print("AUC_nonstandard_change_k: ", AUC_nonstandard_change_k)
print("AUC_standard_change_k: ", AUC_standard_change_k)

plt.figure(figsize=(8, 4))
# compare AUC with/without standardization in changing t (number of features)
plt.subplot(1, 2, 1)
plt.plot(range(1, len(feature_lst_nonstandard_t)+1), AUC_nonstandard_change_t,
         color="r", linestyle="-", marker="*", linewidth=1, label="without standardized")
plt.plot(range(1, len(feature_lst_standard_t)+1), AUC_standard_change_t,
         color="g", linestyle="-", marker="+", linewidth=1, label="with standardized")
plt.title("AUROC with top t features (k=5)")
plt.xlabel("t features")
plt.ylabel("AUROC")
plt.legend(loc='best')

# compare AUC with/without standardization in changing k (number of neighbors)
plt.subplot(1, 2, 2)
plt.plot(range(1, 11), AUC_nonstandard_change_k,
         color="r", linestyle="-", marker="*", linewidth=1, label="without standardized")
plt.plot(range(1, 11), AUC_standard_change_k,
         color="g", linestyle="-", marker="+", linewidth=1, label="with standardized")
plt.title("AUROC with top k neighbors (t=30)")
plt.xlabel("k neighbors")
plt.ylabel("AUROC")
plt.legend(loc='best')

plt.show()
