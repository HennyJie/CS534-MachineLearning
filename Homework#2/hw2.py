'''
@Description: Logistic Regression for Brier Score and Price Predicting, CS534-Machine Learning Hw2
@Author: Hejie Cui
@Date: 2019-09-28 18:35:20
@LastEditTime: 2019-09-30 23:20:57
'''
import datetime
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import ElasticNet
import pandas as pd
from scipy.special import expit
from sklearn.metrics import mean_squared_error

'''
@description: Problem 1
@param {type} 
@return: 
'''
# class LogisticBrier:
#     def __init__(self):
#         self.coef = None

#     # def loss(self, y, X, intercept, coef):
#     #     return np.sum(np.square(y - intercept - expit(np.dot(X, coef))))

#     def fit(self, X, y):

#         # Implement Newton’s method to solve the equation
#         coef_init = np.zeros(np.shape(X)[1])
#         coef_current = coef_init
#         coef_difference = float("inf")
#         difference_threshold = 0.1

#         while np.sum(abs(coef_difference)) > difference_threshold:
#             # Derive the first and second derivative of the loss function
#             t = expit(np.dot(X, coef_current))
#             first_derivative = np.sum(-2 * (y-t) * t * (1-t))
#             print("first_derivative", first_derivative)
#             second_derivative = np.sum(-2 *
#                                        (y - 2*y*t - 2*t + 3*t**2) * t * (1-t))
#             print("second_derivative", second_derivative)

#             coef_next = coef_current - first_derivative / second_derivative
#             coef_difference = coef_next - coef_current
#             coef_current = coef_next

#         self.coef = coef_current

#     def get_coef(self):
#         return self.coef


# data_breast_cancer = load_breast_cancer()
# X = data_breast_cancer.data
# y = data_breast_cancer.target
# # print("data_breast_cancer.data shape: ", np.shape(data_breast_cancer.data))
# # print("data_breast_cancer.target shape: ", np.shape(data_breast_cancer.target))
# logistBrier = LogisticBrier()
# logistBrier.fit(X, y)
# coef = logistBrier.get_coef()

# # Compare the coefficients from the regular Logistic regression and interpret the differences
# coef_brier = pd.DataFrame(
#     {"Feature": data_breast_cancer.feature_names, "Coefficients": np.transpose(coef).flatten()})
# print("coefficients from the minimizing Brier Score Logistic regression: ", coef_brier)

# logistic = LogisticRegression().fit(X, y)
# coef_regular = pd.DataFrame(
#     {"Feature": data_breast_cancer.feature_names, "Coefficients": np.transpose(logistic.coef_).flatten()})
# print("coefficients from the regular Logistic regression: ", coef_regular)


'''
@description: Problem 2
@param {type} 
@return: 
'''
df = pd.read_csv("AAPL.csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df.set_index(df['Date'])
df = df.sort_index()
df_train = df['2014-07-21':'2019-4-30']
df_test = df['2019-5-1':]
# print('Train Dataset shape:', df_train.shape)
# print('Test Dataset shape:', df_test.shape)

# Make various features that may help the predictive algorithms
# adjclose_price = df_train['Adj Close']

high_price = df_train['High']
low_price = df_train['Low']
open_price = df_train['Open']
close_price = df_train['Close']

close_SMA = close_price.rolling(window=20).mean()
close_EMA = close_price.ewm(span=20, adjust=False).mean()
# print("close_SMA: ", close_SMA)
# print("close_EMA: ", close_EMA)
# returns = close_price / close_price.shift(1) - 1
highlow_percentage = (high_price - low_price) / close_price * 100.0
percentage_change = (close_price - open_price) / open_price * 100.0

volume = df_train['Volume']
trade_quantity = volume * close_price

X_train_pd = pd.DataFrame(close_SMA, close_EMA, highlow_percentage,
                          percentage_change, volume, trade_quantity)
print("X_train_pd: ", X_train_pd)
X_train = X_train_pd.to_numpy()

y_train = np.zeros(np.shape(df_train)[0])
y_train[0] = None
for i in range(1, np.shape(df_train)[0]):
    y_train[i] = df_train['Adj Close'][i] - df_train['Adj Close'][i-1]
print("y_train: ", y_train)
# Use Elastic-Net with varying alpha and lambda
# elastic = ElasticNet(random_state=0).fit(X_train, y_train)
# apple_coef = elastic.coef_
# apple_intercept = elastic.intercept_

# Use various evaluation metrics and interpret results
