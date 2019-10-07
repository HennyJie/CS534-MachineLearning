'''
@Description: Test Script for CS534-Machine Learning Hw2
@Author: Hejie Cui
@Date: 2019-10-06 22:49:53
@LastEditTime: 2019-10-06 23:00:00
'''
from hw2 import LogisticBrier
from hw2 import PricePrediction
import datetime
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import ElasticNet, ElasticNetCV
import pandas as pd
from scipy.special import expit
from sklearn import metrics
from math import sqrt
from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from scipy.stats import uniform

############################################ Problem 1 ###################################################
data_breast_cancer = load_breast_cancer()
X = data_breast_cancer.data
y = data_breast_cancer.target

logistBrier = LogisticBrier()
loss_lst = logistBrier.fit(X, y)
intercept, coef = logistBrier.get_value()
print("logistBrier coef: ", coef)
print("logisBrier intercept: ", intercept)
print("loss lst: ", loss_lst)

# Compare the coefficients from the regular Logistic regression and interpret the differences
logistic = LogisticRegression(
    penalty="none", solver='newton-cg', fit_intercept=False, n_jobs=-1).fit(X, y)
print("LogisticRegression coef: ", logistic.coef_)

loss_LogisticRegression = np.sum(
    (y.reshape((X.shape[0], 1)) - expit(np.dot(X, logistic.coef_.T)))**2)
print("loss of LogisticRegression: ", loss_LogisticRegression)
############################################ Problem 1 ###################################################


############################################ Problem 2 ###################################################
df = pd.read_csv("AAPL.csv")
predictor = PricePrediction(df)
regr = predictor.fit(predictor.train_X, predictor.train_y)
print("best alpha: ", regr.alpha_)
print("best l1_ratio: ", regr.l1_ratio_)
print("best coefficients: ", regr.coef_)

y_pred = regr.predict(predictor.test_X)
y_true = predictor.test_y
MSE, RMSE, RSQ, MAE, MAPE, MedAE = predictor.evaluate(y_true, y_pred)
print("MSE: {}, RMSE: {}, RSQ: {}, MAE: {}, MAPE: {}, MedAE: {}".format(
    MSE, RMSE, RSQ, MAE, MAPE, MedAE))

## interpret of my results ##
# I have constructed a 11-dimention feature matrix from the original data, which is the long term and short
# simple
# best alpha:  0.10256410256410256
# best l1_ratio:  0.14141414141414144
# best coefficients:  [ 8.19211115e-03  2.47098327e-03 -4.36819687e-01  1.71917206e-02
#                      -9.46405452e-02  1.30477476e+00  2.97282920e-01  8.77974812e-02
#                       2.76315245e-02  1.51503200e-08 -1.06450824e-10]
# MSE:  0.9309868149978563
# RMSE:  0.9648765801893299
# RSQ:  0.6126145726005878
# MAE:  0.9277922740292522
# MAPE:  72.94668906622289
# MedAE:  0.9886263896496363
############################################ Problem 2 ###################################################
