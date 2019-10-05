'''
@Description: Logistic Regression for Brier Score and Price Predicting, CS534-Machine Learning Hw2
@Author: Hejie Cui
@Date: 2019-09-28 18:35:20
@LastEditTime: 2019-10-04 20:43:36
'''
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
'''
@description: Problem 1
@param {type}
@return:
'''


class LogisticBrier:
    def __init__(self):
        self.intercept = 0
        self.coef = None

    def loss(self, y, X, coef):
        # return np.sum(np.square(y - expit(np.dot(X, coef)))) / X.shape[0]
        return np.sum(np.square(y - expit(np.dot(X, coef))))

    def fit(self, X, y):

        # Implement Newton’s method to solve the equation
        coef_init = np.zeros(np.shape(X)[1])
        coef_current = coef_init
        # coef_difference = float("inf")
        # difference_threshold = 0.01
        loss_lst = []
        for i in range(30):
            p = expit(np.dot(X, coef_current))
            W = np.diag(p * (1-p))
            # first_derivative = -2 * np.dot(np.dot(X.T, W), y-p)
            # print("first_derivative", first_derivative)
            first_derivative = -2 * np.dot(X.T, (y-p) * p * (1-p))
            # print("first_derivative", first_derivative)

            W = np.diag((y - 2*(y+1)*p + 3*p**2) * p * (1-p))
            XWX = np.dot(np.dot(X.T, W), X)
            second_derivative = -2 * XWX
            # print("second_derivative shape", np.shape(second_derivative))

            second_deri_inv = np.linalg.inv(second_derivative)
            coef_current = coef_current - \
                np.dot(second_deri_inv, first_derivative)
            # coef_next = coef_current - \
            #     np.dot(second_deri_inv, first_derivative)

            # coef_difference = coef_next - coef_current
            # coef_current = coef_next
            loss_lst.append(self.loss(y, X, coef_current))

        print("loss_lst: ", loss_lst)

        # while np.sum(abs(coef_difference)) > difference_threshold:
        #     # Derive the first and second derivative of the loss function
        #     p = expit(np.dot(X, coef_current))
        #     print("p shape: ", np.shape(p))
        #     W = np.diag(p * (1-p))
        #     # first_derivative1 = -2 * np.dot(np.dot(X.T, W), y-p)
        #     # print("first_derivative1", first_derivative1)
        #     first_derivative = -2 * np.dot(X.T, (y-p) * p * (1-p))
        #     print("first_derivative", first_derivative)

        #     W = np.diag((y - 2*(y+1)*p + 3*p**2) * p * (1-p))
        #     XWX = np.dot(np.dot(X.T, W), X)
        #     second_derivative = -2 * XWX
        #     print("second_derivative shape", np.shape(second_derivative))

        #     second_deri_inv = np.linalg.inv(second_derivative)
        #     coef_next = coef_current - \
        #         np.dot(second_deri_inv, first_derivative)

        #     coef_difference = coef_next - coef_current
        #     coef_current = coef_next
        #     loss_current = self.loss(y, X, coef_current)
        #     print("loss_current: ", loss_current)

        self.coef = coef_current

    def get_value(self):
        return self.intercept, self.coef


data_breast_cancer = load_breast_cancer()
X = data_breast_cancer.data
y = data_breast_cancer.target
# y = y - y.mean()
y = y - y.mean() + 0.5

logistBrier = LogisticBrier()
logistBrier.fit(X, y)
intercept, coef = logistBrier.get_value()
print("coef: ", coef)

#########################################IRLS Code From Yubin###########################################
# def loss(y, X, beta):
#     ll = y * np.dot(X, beta) - np.log(1 + np.exp(np.dot(X, beta)))
#     return -np.sum(ll)


# data_breast_cancer = load_breast_cancer()
# X = data_breast_cancer.data
# y = data_breast_cancer.target
# beta = np.zeros(m+1)
# loss_lst = []
# loss_lst.append(loss(y, X, beta))
# for i in range(30):
#     p = expit(np.dot(X, beta))
#     W = np.diag(p * (1-p))
#     XWXinv = np.linalg.inv(np.dot(np.dot(X.T, W), X))
#     beta = beta + np.dot(XWXinv, np.dot(X.T, y-p))
#     loss_lst.append(loss(y, X, beta))
# print(loss_lst)

# eps = 1e-5
# beta = np.zeros(m+1)
# loss_lst = []
# loss_lst.append(loss(y, X, beta))
# for i in range(30):
#     p = np.clip(expit(np.dot(X, beta)), eps, 1-eps)
#     W = np.diag(p * (1-p))
#     XWXinv = np.linalg.inv(np.dot(np.dot(X.T, W), X))
#     beta = beta + np.dot(XWXinv, np.dot(X.T, y-p))
#     loss_lst.append(loss(y, X, beta))
# print(loss_lst)
###########################################################################################

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


class PricePrediction:
    def __init__(self, df):
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index(df['Date'])
        df = df.sort_index()
        df_train = df['2014-07-21':'2019-4-30']
        df_test = df['2019-5-1':]
        window = 20
        self.train_X = self.feature_construct(df_train, window)
        self.train_y = self.label_construct(df_train, window)
        self.test_X = self.feature_construct(df_test, window)
        self.test_y = self.label_construct(df_test, window)

    def feature_construct(self, df, window):
        # Make various features that may help the predictive algorithms
        high_price = df['High']
        low_price = df['Low']
        open_price = df['Open']
        close_price = df['Close']

        close_SMA = close_price.rolling(window=window).mean()
        close_EMA = close_price.ewm(span=window, adjust=False).mean()

        # returns = close_price / close_price.shift(1) - 1
        highlow_percentage = (high_price - low_price) / close_price * 100.0
        percentage_change = (close_price - open_price) / open_price * 100.0

        open = df['Open']
        high = df['High']
        low = df['Low']
        volume = df['Volume']
        trade_quantity = volume * close_price

        X_pd = pd.DataFrame({'close_SMA': close_SMA, 'close_EMA': close_EMA, 'highlow_percentage': highlow_percentage,
                             'percentage_change': percentage_change, 'open': open, 'high': high, 'low': low, 'volume': volume, 'trade_quantity': trade_quantity})
        # X_pd = pd.DataFrame({'close_SMA': close_SMA, 'open': open,
                            #  'high': high, 'low': low, 'volume': volume})

        X_pd = X_pd[window-1:]
        features = X_pd.to_numpy()
        # print("X: ", X)
        # print("X shape: ", np.shape(X))
        return features

    def label_construct(self, df, window):
        labels = np.zeros(np.shape(df)[0])
        labels[0] = 0
        for i in range(1, np.shape(df)[0]):
            labels[i] = df['Close'][i] - df['Close'][i-1]
        # print("y_train: ", y_train)
        # print("y_train shape: ", np.shape(y_train))
        labels = labels[window-1:]
        return labels

    def fit(self, X, y):
        # Use Elastic-Net with varying alpha and lambda
        elastic = ElasticNet()
        param_grid = {
            'alpha': np.linspace(0, 2, num=20),
            'l1_ratio': np.linspace(0, 1, num=100),
        }
        clf = GridSearchCV(elastic, param_grid, cv=5,
                    scoring='neg_log_loss', n_jobs=-1)
        # alpha = uniform(loc=0, scale=2)
        # l1_ratio = uniform(loc=0, scale=1)
        # hyperparameters = dict(alpha=alpha, l1_ratio=l1_ratio)
        # clf = RandomizedSearchCV(
        #     elastic, hyperparameters, random_state=100, n_iter=1000, cv=5, verbose=0, n_jobs=1)

        self.best_model = clf.fit(X, y)
        print("best alpha: ",
              self.best_model.best_estimator_.get_params()['alpha'])
        print("best l1_ratio: ",
              self.best_model.best_estimator_.get_params()['l1_ratio'])

        # alphas = [0.0001, 0.0005, 0.001, 0.01, 0.1, 1, 10]
        # l1_ratio = [0.001, 0.01, 0.05, 0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1]
        # alphas = np.linspace(0, 2, num=20)
        # print(alphas)
        # l1_ratio = np.linspace(0, 1, num=100)
        # print(l1_ratio)
        # regr = ElasticNetCV(l1_ratio=l1_ratio, alphas=alphas,
        #                     cv=5, random_state=0)
        # regr.fit(X, y)
        # print("best alpha: ", regr.alpha_)
        # print("best l1_ratio: ", regr.l1_ratio_)
        # print("best coefficients: ", regr.coef_)
        # return regr

    def predict(self, X):
        # y_pred = regr.predict(X)
        y_pred = self.best_model.predict(X)
        return y_pred

    def evaluate(self, y_true, y_pred):
        MSE = metrics.mean_squared_error(y_true, y_pred)
        print("MSE: ", MSE)
        RMSE = np.sqrt(MSE)
        print("RMSE: ", RMSE)
        RSQ = metrics.r2_score(y_true, y_pred)
        print("RSQ: ", RSQ)
        MAE = metrics.mean_absolute_error(y_true, y_pred)
        print("MAE: ", MAE)
        MAPE = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        print("MAPE: ", MAPE)
        MedAE = metrics.median_absolute_error(y_true, y_pred)
        print("MedAE: ", MedAE)
        MSLE = metrics.mean_squared_log_error(y_true, y_pred)
        print("MSLE: ", MSLE)


df = pd.read_csv("AAPL.csv")
predictor = PricePrediction(df)
predictor.fit(predictor.train_X, predictor.train_y)
y_pred = predictor.predict(predictor.test_X)
y_true = predictor.test_y
predictor.evaluate(y_true, y_pred)
