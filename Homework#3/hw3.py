'''
@Description: In User Settings Edit
@Author: your name
@Date: 2019-10-08 15:46:41
@LastEditTime: 2019-10-18 23:10:51
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
    def __init__(self, max_depth=2):
        self.max_depth = max_depth
        self.tree = None

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
        N1_index = np.logical_and(feature1 <= value1, feature2 <= value2)
        N2_index = np.logical_and(feature1 <= value1, feature2 > value2)
        N3_index = np.logical_and(feature1 > value1, feature2 <= value2)
        N4_index = np.logical_and(feature1 > value1, feature2 > value2)

        target_N1 = target[N1_index]
        target_N2 = target[N2_index]
        target_N3 = target[N3_index]
        target_N4 = target[N4_index]

        # p_N1 = target_N1.size / target.size
        # p_N2 = target_N2.size / target.size
        # p_N3 = target_N3.size / target.size
        # p_N4 = target_N4.size / target.size

        entropy = target_N1.size * self.get_entropy(target_N1) + \
            target_N2.size * self.get_entropy(target_N2) + \
            target_N3.size * self.get_entropy(target_N3) + \
            target_N4.size * self.get_entropy(target_N4)

        return entropy

    def generate_split_values(self, feature, target):
        split_interval = 50
        argsort = feature.argsort()
        f1 = feature[argsort]
        split_values = []

        for i in range(0, f1.shape[0], split_interval):
            split_values.append(feature[i])

        return np.array(split_values)

    def get_features_pair_min_entropy(self, feature1, feature2, target):
        min_entropy = float('inf')
        min_S1 = 0
        min_S2 = 0
        values1 = self.generate_split_values(feature1, target)
        values2 = self.generate_split_values(feature2, target)

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
                    S1, S2, entropy = self.get_features_pair_min_entropy(
                        X[:, i], X[:, j], y)

                    if entropy < min_entropy:
                        min_entropy = entropy
                        min_X1 = i
                        min_S1 = S1
                        min_X2 = j
                        min_S2 = S2

        return min_X1, min_S1, min_X2, min_S2

    def fit(self, X, y):
        self.tree = self.decision_tree(X, y, self.max_depth)

    def decision_tree(self, X, y, max_depth):
        n, m = X.shape
        if n < 3 or max_depth == 0:
            return round(np.mean(y))

        X1, S1, X2, S2 = self.select_split_pairs(X, y)
        # print("Decision Tree: X1:{}, S1:{}, X2:{}, S2:{}".format(X1, S1, X2, S2))

        N1_index = np.logical_and(X[:, X1] <= S1, X[:, X2] <= S2)
        N2_index = np.logical_and(X[:, X1] <= S1, X[:, X2] > S2)
        N3_index = np.logical_and(X[:, X1] > S1, X[:, X2] <= S2)
        N4_index = np.logical_and(X[:, X1] > S1, X[:, X2] > S2)

        X_N1, y_N1 = X[N1_index, :], y[N1_index]
        X_N2, y_N2 = X[N2_index, :], y[N2_index]
        X_N3, y_N3 = X[N3_index, :], y[N3_index]
        X_N4, y_N4 = X[N4_index, :], y[N4_index]

        return {'split_var_1': X1,
                'split_value_1': S1,
                'split_var_2': X2,
                'split_value_2': S2,
                'N1': self.decision_tree(X_N1, y_N1, max_depth-1),
                'N2': self.decision_tree(X_N2, y_N2, max_depth-1),
                'N3': self.decision_tree(X_N3, y_N3, max_depth-1),
                'N4': self.decision_tree(X_N4, y_N4, max_depth-1)}

    def predict(self, X):
        y_pred = []
        for x in X:
            y_pred.append(self.single_sample_predict(x, self.tree))
        return y_pred

    def single_sample_predict(self, x, tree):

        if not isinstance(tree, dict):
            return tree

        if x[tree['split_var_1']] <= tree['split_value_1'] and x[tree['split_var_2']] <= tree['split_value_2']:
            return self.single_sample_predict(x, tree['N1'])
        elif x[tree['split_var_1']] <= tree['split_value_1'] and x[tree['split_var_2']] > tree['split_value_2']:
            return self.single_sample_predict(x, tree['N2'])
        elif x[tree['split_var_1']] > tree['split_value_1'] and x[tree['split_var_2']] <= tree['split_value_2']:
            return self.single_sample_predict(x, tree['N3'])
        elif x[tree['split_var_1']] > tree['split_value_1'] and x[tree['split_var_2']] > tree['split_value_2']:
            return self.single_sample_predict(x, tree['N4'])


# data_breast_cancer = load_breast_cancer()
# feature_names = data_breast_cancer.feature_names
# class_names = data_breast_cancer.target_names

# X = data_breast_cancer.data
# y = data_breast_cancer.target
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.3, random_state=100)
# clf = QuarternaryDecisionTree()
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# print("==================Problem 1====================")
# print("Accuracy: ", accuracy_score(y_test, y_pred))
# print("Cofusion Matrix: ", confusion_matrix(y_test, y_pred))
# print("clf.tree: ", clf.tree)


class DaRDecisionTree:
    def __init__(self, max_depth=2):
        self.max_depth = max_depth
        self.tree = None

    def get_condition_MSE(self, feature, value, target):
        L_index = feature <= value
        R_index = feature > value

        target_L = target[L_index]
        target_R = target[R_index]

        MSE = target_L.size * np.var(target_L) + \
            target_R.size * np.var(target_R)

        return MSE

    def get_feature_min_MSE(self, feature, target):
        min_MSE = float('inf')
        min_S = 0
        argsort = feature.argsort()
        f1 = feature[argsort]

        for v in f1:
            MSE = self.get_condition_MSE(
                feature, v, target)
            if MSE < min_MSE:
                min_MSE = MSE
                min_S = v

        return min_S, min_MSE

    def select_split_pairs(self, X, y):
        min_MSE = float('inf')
        features_num = X.shape[1]
        min_X1 = None
        min_X2 = None
        min_S1 = 0
        min_S2 = 0

        for i in range(features_num):
            S, MSE = self.get_feature_min_MSE(X[:, i], y)

            if MSE < min_MSE:
                min_MSE = MSE
                min_X = i
                min_S = S

        return min_X, min_S

    def fit(self, X, y):
        self.tree = self.decision_tree(X, y, self.max_depth)

    def decision_tree(self, X, y, max_depth):
        n, m = X.shape
        if n < 3 or max_depth == 0:
            r = Ridge()
            r.fit(X, y)
            return r

        j_best, s_value = self.select_split_pairs(X, y)

        L_index = X[:, j_best] <= s_value
        R_index = X[:, j_best] > s_value

        X_L, y_L = X[L_index, :], y[L_index]
        X_R, y_R = X[R_index, :], y[R_index]

        return {'split_var': j_best,
                'split_value': s_value,
                'L': self.decision_tree(X_L, y_L, max_depth-1),
                'R': self.decision_tree(X_R, y_R, max_depth-1)}

    def predict(self, X):
        y_pred = []
        for x in X:
            y_pred.append(self.single_sample_predict(x, self.tree))
        return y_pred

    def single_sample_predict(self, x, tree):

        if not isinstance(tree, dict):
            return tree.predict(x.reshape(1, -1))[0]

        if x[tree['split_var']] <= tree['split_value']:
            return self.single_sample_predict(x, tree['L'])
        elif x[tree['split_var']] > tree['split_value']:
            return self.single_sample_predict(x, tree['R'])


data_boston = load_boston()
X = data_boston.data
y = data_boston.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=100)

clf = DaRDecisionTree()
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)

# MSE = metrics.mean_squared_error(y_test, y_pred)
# RMSE = np.sqrt(MSE)
# RSQ = metrics.r2_score(y_test, y_pred)
# MAE = metrics.mean_absolute_error(y_test, y_pred)
# MAPE = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
# MedAE = metrics.median_absolute_error(y_test, y_pred)
print("==================Problem 2====================")
# print("MSE: {}, RMSE: {}, RSQ: {}, MAE: {}, MAPE: {}, MedAE: {}".format(
#     MSE, RMSE, RSQ, MAE, MAPE, MedAE))

clf.fit(X, y)
y_pred = clf.predict(X)
mse = metrics.mean_squared_error(y, y_pred)
print("mse: ", mse)
