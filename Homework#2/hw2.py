'''
@Description: Logistic Regression for Brier Score and Price Predicting, CS534-Machine Learning Hw2
@Author: Hejie Cui
@Date: 2019-09-28 18:35:20
@LastEditTime: 2019-09-30 16:36:18
'''
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import ElasticNet
import pandas as pd
from scipy.special import expit

'''
@description: Problem 1
@param {type} 
@return: 
'''


class LogisticBrier:
    def __init__(self):
        self.coef = None

    # def loss(self, y, X, intercept, coef):
    #     return np.sum(np.square(y - intercept - expit(np.dot(X, coef))))

    def fit(self, X, y):

        # Implement Newton’s method to solve the equation
        coef_init = np.zeros(np.shape(X)[1])
        coef_current = coef_init
        coef_difference = float("inf")
        difference_threshold = 0.1

        while np.sum(abs(coef_difference)) > difference_threshold:
            # Derive the first and second derivative of the loss function
            t = expit(np.dot(X, coef_current))
            first_derivative = np.sum(-2 * (y-t) * t * (1-t))
            print("first_derivative", first_derivative)
            second_derivative = np.sum(-2 *
                                       (y - 2*y*t - 2*t + 3*t**2) * t * (1-t))
            print("second_derivative", second_derivative)

            coef_next = coef_current - first_derivative / second_derivative
            coef_difference = coef_next - coef_current
            coef_current = coef_next

        self.coef = coef_current

    def get_coef(self):
        return self.coef


data_breast_cancer = load_breast_cancer()
X = data_breast_cancer.data
y = data_breast_cancer.target
# print("data_breast_cancer.data shape: ", np.shape(data_breast_cancer.data))
# print("data_breast_cancer.target shape: ", np.shape(data_breast_cancer.target))
logistBrier = LogisticBrier()
logistBrier.fit(X, y)
coef = logistBrier.get_coef()

# Compare the coefficients from the regular Logistic regression and interpret the differences
coef_brier = pd.DataFrame(
    {"Feature": data_breast_cancer.feature_names, "Coefficients": np.transpose(coef).flatten()})
print("coefficients from the minimizing Brier Score Logistic regression: ", coef_brier)

logistic = LogisticRegression().fit(X, y)
coef_regular = pd.DataFrame(
    {"Feature": data_breast_cancer.feature_names, "Coefficients": np.transpose(logistic.coef_).flatten()})
print("coefficients from the regular Logistic regression: ", coef_regular)


'''
@description: Problem 2
@param {type} 
@return: 
'''
data_apple = pd.read_csv("AAPL.csv")
# close_price = data_apple['Close'].to_numpy()

# # Make various features that may help the predictive algorithms
# close_SMA = close_price.rolling(window=20).mean()
# close_EMA = close_price.ewm(span=20, adjust=False).mean()

# # Use Elastic-Net with varying alpha and lambda
# elastic = ElasticNet(random_state=0).fit(X, y)
# apple_coef = elastic.coef_
# apple_intercept = elastic.intercept_

# Use various evaluation metrics and interpret results
