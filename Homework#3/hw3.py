'''
@Description: In User Settings Edit
@Author: your name
@Date: 2019-10-08 15:46:41
@LastEditTime: 2019-10-18 16:40:31
@LastEditors: Please set LastEditors
'''
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_boston
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Ridge
from math import log


class QuarternaryDecisionTree:
    def __init__(self, max_depth):
        self.max_depth = max_depth

    def get_entropy(self, target):
        label = np.unique(target)
        n = label.size
        count = np.zeros(n)
        p_i = np.zeros(n)

        for i in range(n):
            count[i] = target[target == label[i]].size

        p_i = np.divide(count, target.size)
        entropy = 0

        for i in range(n):
            entropy = entropy - p_i[i] * np.log2(p_i[i])

        return entropy

    def get_condition_entropy(self, feature1, value1, feature2, value2, target):
        N1_index, N2_index, N3_index, N4_index = [], [], [], []
        for i in range(feature1.shape[0]):
            if feature1[i] <= value1 and feature2[i] <= value2:
                N1_index.append(i)
            elif feature1[i] <= value1 and feature2[i] > value2:
                N2_index.append(i)
            elif feature1[i] > value1 and feature2[i] <= value2:
                N3_index.append(i)
            elif feature1[i] > value1 and feature2[i] > value2:
                N4_index.append(i)

        target_N1 = target[N1_index]
        target_N2 = target[N2_index]
        target_N3 = target[N3_index]
        target_N4 = target[N4_index]

        p_N1 = target_N1.size / target.size
        p_N2 = target_N2.size / target.size
        p_N3 = target_N3.size / target.size
        p_N4 = target_N4.size / target.size

        entropy = p_N1 * self.get_entropy(target_N1) + \
            p_N2 * self.get_entropy(target_N2) + \
            p_N3 * self.get_entropy(target_N3) + \
            p_N4 * self.get_entropy(target_N4)

        return entropy

    def generate_split_values(self, feature, target):
        argsort = feature.argsort()
        # print("argsort: ", argsort)

        f1 = feature[argsort]
        print("f1: ", f1)

        t1 = target[argsort]
        last_value = target[0]
        split_values = []

        for i in range(t1.size):
            if last_value != t1[i]:
                split_values.append((f1[i] + f1[i-1])/2)
                last_value = t1[i]

        return np.array(split_values)

    def get_features_pair_entropy(self, feature1, feature2, target):
        min_entropy = float('inf')
        min_S1 = 0
        min_S2 = 0
        values1 = self.generate_split_values(feature1, target)
        # print("feature1: ", feature1)
        print("values1: ", values1)
        values2 = self.generate_split_values(feature2, target)
        # print("feature2: ", feature2)
        print("values2: ", values2)

        for v1 in values1:
            for v2 in values2:
                entropy = self.get_condition_entropy(
                    feature1, v1, feature2, v2, target)
                if entropy < min_entropy:
                    min_entropy = entropy
                    min_S1 = v1
                    min_S2 = v2

        return min_S1, min_S2, min_entropy

    def select_split_pairs(self, X, y):
        min_entropy = float('inf')
        features_num = X.shape[1]
        min_X1 = None
        min_X2 = None
        min_S1 = 0
        min_S2 = 0

        for i in range(features_num):
            for j in range(features_num):
                if i == j:
                    continue
                else:
                    S1, S2, entropy = self.get_features_pair_entropy(
                        X[:, i], X[:, j], y)
                    print("entropy: ", entropy)
                    if entropy < min_entropy:
                        min_entropy = entropy
                        min_X1 = i
                        min_S1 = S1
                        min_X2 = j
                        min_S2 = S2

        return min_X1, min_S1, min_X2, min_S2, min_entropy

    def fit(self, X, y):
        # num_samples, num_features = X.shape
        tree = self.decision_tree(X, y, 2)
        return tree

    def decision_tree(self, X, y, max_depth):
        n, m = X.shape
        if n < 3 or max_depth == 0:
            return np.mean(y)

        X1, S1, X2, S2, entropy = self.select_split_pairs(X, y)
        print("X1:{}, S1:{}, X2:{}, S2:{}".format(X1, S1, X2, S2))

        N1_index = X[:, X1] <= S1 and X[:, X2] <= S2
        N2_index = X[:, X1] <= S1 and X[:, X2] > S2
        N3_index = X[:, X1] > S1 and X[:, X2] <= S2
        N4_index = X[:, X1] > S1 and X[:, X2] > S2

        X_N1, y_N1 = X[N1_index, :], y[N1_index]
        X_N2, y_N2 = X[N2_index, :], y[N2_index]
        X_N3, y_N3 = X[N3_index, :], y[N3_index]
        X_N4, y_N4 = X[N4_index, :], y[N4_index]

        return {"split_var 1: ", X_N1,
                "split_value 1: ", S1,
                "split_var 2: ", X_N2,
                "split_value 2: ", S2,
                "N1: ", decision_tree(X_N1, y_N1, max_depth-1),
                "N2: ", decision_tree(X_N2, y_N2, max_depth-1),
                "N3: ", decision_tree(X_N3, y_N3, max_depth-1),
                "N4: ", decision_tree(X_N4, y_N4, max_depth-1)}

    def predict(self, X):

        return y


data_breast_cancer = load_breast_cancer()
feature_names = data_breast_cancer.feature_names
class_names = data_breast_cancer.target_names
# print(feature_names)
# print(class_names)

X = data_breast_cancer.data
y = data_breast_cancer.target
# print("X shape: ", X.shape)
# print("y shape: ", y.shape)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=100)
# clf = DecisionTreeClassifier()
clf = QuarternaryDecisionTree()
clf = clf.fit(X_train, y_train)
print(clf)
# y_pred = clf.predict(X_test)
# print("Accuracy: ", accuracy_score(y_test, y_pred))
# print("Cofusion Matrix: ", confusion_matrix(y_test, y_pred))


# class DaRDecisionTree:
#     def __init__(self):

#     def fit(self, X, y):

#     def predict(self, X):


# data_boston = load_boston()
# X = data_boston.data
# y = data_boston.target
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.3, random_state=100)
# print("X: ", X)
# print("y: ", y)
# clf = DecisionTreeRegressor()
# clf = DaRDecisionTree()
# clf = clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)

# MSE = metrics.mean_squared_error(y_true, y_pred)
# RMSE = np.sqrt(MSE)
# RSQ = metrics.r2_score(y_true, y_pred)
# MAE = metrics.mean_absolute_error(y_true, y_pred)
# MAPE = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
# MedAE = metrics.median_absolute_error(y_true, y_pred)
# print("MSE: {}, RMSE: {}, RSQ: {}, MAE: {}, MAPE: {}, MedAE: {}".format(
# MSE, RMSE, RSQ, MAE, MAPE, MedAE))
