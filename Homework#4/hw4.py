'''
@Author: Hejie Cui
@Date: 2019-11-09 19:31:31
@LastEditTime: 2019-11-17 23:13:56
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
    X, y, test_size=0.3, random_state=1024)
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
k_values = [1, 5, 10, 20, 50, 100, 250, X_train.shape[0]]
AUC_nonstandard_change_k_full = np.zeros(len(k_values))
AUC_nonstandard_change_k_best = np.zeros(len(k_values))
AUC_standard_change_k_full = np.zeros(len(k_values))
AUC_standard_change_k_best = np.zeros(len(k_values))

for index, k in enumerate(k_values):
    # change k, without features standardized (all features)
    feature_lst_nonstandard_k = knn.get_feature_order(
        X_train, y_train, X_test, k)
    y_pred = knn.kNNpredict(X_train[:, feature_lst_nonstandard_k], y_train,
                            X_test[:, feature_lst_nonstandard_k], k, train_status=False)
    AUC_nonstandard_change_k_full[index] = metrics.roc_auc_score(
        y_test, y_pred)

    # change k, without features standardized (features achieved max auroc)
    max_auroc = 0.0
    best_t = -1
    for t in range(1, len(feature_lst_nonstandard_k)+1):
        y_pred = knn.kNNpredict(X_train[:, feature_lst_nonstandard_k[:t]], y_train,
                                X_test[:, feature_lst_nonstandard_k[:t]], k, train_status=False)
        auroc = metrics.roc_auc_score(y_test, y_pred)
        if auroc > max_auroc:
            max_auroc = auroc
            best_t = t
    y_pred = knn.kNNpredict(X_train[:, feature_lst_nonstandard_k[:best_t]], y_train,
                            X_test[:, feature_lst_nonstandard_k[:best_t]], k, train_status=False)
    AUC_nonstandard_change_k_best[index] = metrics.roc_auc_score(
        y_test, y_pred)

    # change k, with features standardized (all features)
    feature_lst_standard_k = knn.get_feature_order(
        X_train_standardized, y_train, X_test_standardized, k)
    y_pred = knn.kNNpredict(
        X_train_standardized[:, feature_lst_standard_k], y_train, X_test_standardized[:, feature_lst_standard_k], k, train_status=False)
    AUC_standard_change_k_full[index] = metrics.roc_auc_score(y_test, y_pred)

    # change k, with features standardized (features achieved max auroc)
    max_auroc = 0.0
    best_t = -1
    for t in range(1, len(feature_lst_standard_k)+1):
        y_pred = knn.kNNpredict(X_train_standardized[:, feature_lst_standard_t[:t]], y_train,
                                X_test_standardized[:, feature_lst_standard_t[:t]], k, train_status=False)
        auroc = metrics.roc_auc_score(y_test, y_pred)
        if auroc > max_auroc:
            max_auroc = auroc
            best_t = t
    y_pred = knn.kNNpredict(X_train_standardized[:, feature_lst_standard_k[:best_t]], y_train,
                            X_test_standardized[:, feature_lst_standard_k[:best_t]], k, train_status=False)
    AUC_standard_change_k_best[index] = metrics.roc_auc_score(
        y_test, y_pred)
print("AUC_nonstandard_change_k_full: ", AUC_nonstandard_change_k_full)
print("AUC_nonstandard_change_k_best: ", AUC_nonstandard_change_k_best)
print("AUC_standard_change_k_full: ", AUC_standard_change_k_full)
print("AUC_standard_change_k_best: ", AUC_standard_change_k_best)

plt.figure(figsize=(12, 3.6))
# compare AUC with/without standardization in changing the number of features t (k=5)
plt.subplot(131)
plt.plot(range(1, len(feature_lst_nonstandard_t)+1), AUC_nonstandard_change_t,
         color="r", linestyle="-", marker="*", linewidth=1, label="without standardized")
plt.plot(range(1, len(feature_lst_standard_t)+1), AUC_standard_change_t,
         color="g", linestyle="-", marker="+", linewidth=1, label="with standardized")
plt.title("AUROC for top t features \n (k=5)")
plt.xlabel("t")
plt.ylabel("AUROC")
plt.legend(loc='best')

# compare AUC with/without standardization in changing the number of neighbors k (t=30)
plt.subplot(132)
plt.plot(k_values, AUC_nonstandard_change_k_full,
         color="r", linestyle="-", marker="*", linewidth=1, label="without standardized")
plt.plot(k_values, AUC_standard_change_k_full,
         color="g", linestyle="-", marker="+", linewidth=1, label="with standardized")
plt.title("AUROC for top k neighbors \n (all features)")
plt.xlabel("k")
plt.ylabel("AUROC")
plt.legend(loc='best')

# # compare AUC with/without standardization in changing the number of neighbors k
# (t= the number of features that reached highest AUROC)
plt.subplot(133)
plt.plot(k_values, AUC_nonstandard_change_k_best,
         color="r", linestyle="-", marker="*", linewidth=1, label="without standardized")
plt.plot(k_values, AUC_standard_change_k_best,
         color="g", linestyle="-", marker="+", linewidth=1, label="with standardized")
plt.title("AUROC for top k neighbors \n (features achieved max auroc)")
plt.xlabel("k")
plt.ylabel("AUROC")
plt.legend(loc='best')

plt.tight_layout()
plt.show()
