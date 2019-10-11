'''
@Description: In User Settings Edit
@Author: your name
@Date: 2019-10-08 15:46:41
@LastEditTime: 2019-10-10 00:03:53
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


# class QuarternaryDecisionTree:
#     def __init__(self):

#     def fit(self, X, y):
#         n_samples, self.n_features_ = X.shape

#         return self

#     def predict(self, X):

#         return y


data_breast_cancer = load_breast_cancer()
print("data_breast_cancer: ", data_breast_cancer)
# print("data shape:", np.shape(data_breast_cancer))

X = data_breast_cancer.data
y = data_breast_cancer.target
print("X shape: ", X.shape)
print("y shape: ", y.shape)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=100)
# clf = DecisionTreeClassifier()
# clf = QuarternaryDecisionTree()
# clf = clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# print("Accuracy: ", accuracy_score(y_test, y_pred))
# print("Cofusion Matrix: ", confusion_matrix(y_test, y_pred))


# class DaRDecisionTree:
#     def __init__(self):

#     def fit(self, X, y):

#     def predict(self, X):


data_boston = load_boston()
X = data_boston.data
y = data_boston.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=100)
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
