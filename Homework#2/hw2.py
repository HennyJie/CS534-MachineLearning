'''
@Description: Logistic Regression for Brier Score and Price Predicting, CS534-Machine Learning Hw2
@Author: Hejie Cui
@Date: 2019-09-28 18:35:20
@LastEditTime: 2019-09-29 16:08:29
'''
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import ElasticNet
import pandas as pd

'''
@description: Problem 1
@param {type} 
@return: 
'''


def loss(y, X, intercept, coef):
    return np.sum(np.square(y - intercept - expit(np.dot(X, coef))))


data_breast_cancer = load_breast_cancer()
X = data_breast_cancer.data
y = data_breast_cancer.target
# print("data_breast_cancer.data shape: ", np.shape(data_breast_cancer.data))
# print("data_breast_cancer.target shape: ", np.shape(data_breast_cancer.target))

logistic = LogisticRegression().fit(X, y)
coefficients = pd.DataFrame(
    {"Feature": data_breast_cancer.feature_names, "Coefficients": np.transpose(logistic.coef_).flatten()})
print("coefficients: ", coefficients)


'''
@description: Problem 2
@param {type} 
@return: 
'''
data_apple = pd.read_csv("AAPL.csv")
close_price = data_apple['Close'].to_numpy()

# Make various features that may help the predictive algorithms
close_SMA = close_price.rolling(window=20).mean()
close_EMA = close_price.ewm(span=20, adjust=False).mean()

# Use Elastic-Net with varying alpha and lambda
elastic = ElasticNet(random_state=0).fit(X, y)
apple_coef = elastic.coef_
apple_intercept = elastic.intercept_

# Use various evaluation metrics and interpret results
