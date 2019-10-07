'''
@Description: Logistic Regression for Brier Score and Price Predicting, CS534-Machine Learning Hw2
@Author: Hejie Cui
@Date: 2019-09-28 18:35:20
@LastEditTime: 2019-10-06 23:03:05
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


class LogisticBrier:
    def __init__(self):
        self.intercept = 0
        self.coef = None

    def loss(self, y, X, coef):
        return np.sum(np.square(y - self.intercept - expit(np.dot(X, coef))))

    # The iterated process to calculate self.intercept and self.coef.
    # My strategy is first fix self.beta as 0, then calculate self.coef,
    # then fix self.coef and calculate self.intercept and so on...
    def fit(self, X, y):
        # Implement Newton’s method to solve the equation
        n, m = X.shape
        self.coef = np.zeros(m)
        loss_lst = []
        max_iteration = 100

        for i in range(max_iteration):
            coef_old = self.coef
            p = expit(np.dot(X, coef_old))

            # Derive the first derivative of the loss function
            first_derivative = -2 * \
                np.dot(X.T, (y - self.intercept - p) * p * (1-p))
            # Terminate condition of Newton's method
            if np.abs(first_derivative).mean() < 1e-3:
                break

            # Derive the second derivative(Hessian Metrix) of the loss function
            W = np.diag(-2 * (y-self.intercept - 2*(y-self.intercept+1)
                              * p + 3*p**2) * p * (1-p))
            Hessian = np.dot(np.dot(X.T, W), X)

            # Use Newton’s method to update coef
            Hessian_inv = np.linalg.inv(Hessian)
            self.coef = coef_old - np.dot(Hessian_inv, first_derivative)
            # Use coef to update intercept
            self.intercept = y.mean() - np.mean(expit(np.dot(X, self.coef)))

            current_loss = self.loss(y, X, self.coef)
            loss_lst.append(current_loss)

        return loss_lst
        # print("loss_lst: ", loss_lst)

    def get_value(self):
        return self.intercept, self.coef


class PricePrediction:
    def __init__(self, df):
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index(df['Date'])
        df = df.sort_index()
        df_train = df['2014-07-21':'2019-4-30']
        df_test = df['2019-5-1':]

        # Devide the data as training set and test set, max window is used for calculating moving average
        max_window = 50
        self.train_X = self.feature_construct(df_train, max_window)
        self.train_y = self.label_construct(df_train, max_window)
        self.test_X = self.feature_construct(df_test, max_window)
        self.test_y = self.label_construct(df_test, max_window)

    def feature_construct(self, df, window):
        # Make various features that may help the predictive algorithms
        high_price = df['High']
        low_price = df['Low']
        open_price = df['Open']
        close_price = df['Close']

        # Long term simple moving average and exponent moving avearge
        close_SMA_long = close_price.rolling(window=window).mean()
        close_EMA_long = close_price.ewm(span=window, adjust=False).mean()
        # Short term simple moving average and exponent moving avearge
        close_SMA_short = close_price.rolling(window=int(window*0.1)).mean()
        close_EMA_short = close_price.ewm(
            span=int(window*0.1), adjust=False).mean()

        # Two features related to percentage
        highlow_percentage = (high_price - low_price) / close_price * 100.0
        percentage_change = (close_price - open_price) / open_price * 100.0

        open = df['Open']
        high = df['High']
        low = df['Low']
        volume = df['Volume']
        trade_quantity = volume * close_price

        # Combined the above columns to get the feature metrix
        X_pd = pd.DataFrame({'close_SMA_long': close_SMA_long, 'close_EMA_long': close_EMA_long,
                             'close_SMA_short': close_SMA_short, 'close_EMA_short': close_EMA_short,
                             'highlow_percentage': highlow_percentage, 'percentage_change': percentage_change,
                             'open': open, 'high': high, 'low': low, 'volume': volume, 'trade_quantity': trade_quantity})

        X_pd = X_pd[window-1:]
        features = X_pd.to_numpy()
        return features

    def label_construct(self, df, window):
        # labels is the residuals of two continuous days calculated from the close price column
        labels = np.zeros(np.shape(df)[0])
        labels[0] = 0
        for i in range(1, np.shape(df)[0]):
            labels[i] = df['Close'][i] - df['Close'][i-1]
        labels = labels[window-1:]
        return labels

    def fit(self, X, y):
        # Use Elastic-Net with varying alpha and lambda

        # elastic = ElasticNet()
        # param_grid = {
        #     'alpha': np.linspace(0, 2, num=40),
        #     'l1_ratio': np.linspace(0, 1, num=100),
        # }
        # clf = GridSearchCV(elastic, param_grid, cv=5, n_jobs=-1)

        # alpha = uniform(loc=0, scale=2)
        # l1_ratio = uniform(loc=0, scale=1)
        # hyperparameters = dict(alpha=alpha, l1_ratio=l1_ratio)
        # clf = RandomizedSearchCV(
        #     elastic, hyperparameters, random_state=100, n_iter=1000, cv=5, verbose=0, n_jobs=1)

        # regr = clf.fit(X, y)
        # print("best alpha: ",
        #       regr.best_estimator_.get_params()['alpha'])
        # print("best l1_ratio: ",
        #       regr.best_estimator_.get_params()['l1_ratio'])

        alphas = np.linspace(0, 2, num=40)
        l1_ratio = np.linspace(0, 1, num=100)
        regr = ElasticNetCV(l1_ratio=l1_ratio, alphas=alphas,
                            cv=5, random_state=0)
        regr.fit(X, y)

        return regr

    # Various evaluation metrics
    def evaluate(self, y_true, y_pred):
        MSE = metrics.mean_squared_error(y_true, y_pred)
        RMSE = np.sqrt(MSE)
        RSQ = metrics.r2_score(y_true, y_pred)
        MAE = metrics.mean_absolute_error(y_true, y_pred)
        MAPE = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        MedAE = metrics.median_absolute_error(y_true, y_pred)

        return MSE, RMSE, RSQ, MAE, MAPE, MedAE


############################################ Problem 1 ###################################################
data_breast_cancer = load_breast_cancer()
X = data_breast_cancer.data
y = data_breast_cancer.target

# The new logistic regression for brier score
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
### interpret of my results ###

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

### interpret of my results ###
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
