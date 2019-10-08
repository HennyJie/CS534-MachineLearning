'''
@Description: Logistic Regression for Brier Score and Price Predicting, CS534-Machine Learning Hw2
@Author: Hejie Cui
@Date: 2019-09-28 18:35:20
@LastEditTime: 2019-10-07 22:34:07
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
from scipy.ndimage.interpolation import shift
import matplotlib
import matplotlib.pyplot as plt


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
        self.train_X = self.feature_construct(df_train)
        self.train_y = self.label_construct(df_train)
        self.test_X = self.feature_construct(df_test)
        self.test_y = self.label_construct(df_test)
        self.close_test = df_test['Close'][1:-1].to_numpy()

    def moving_average(self, X, window_size, type):
        n = X.shape[0]
        moving_average = np.zeros(n)

        for i in range(1, n):
            if type == "simple":
                if i > window_size:
                    moving_average[i-1] = np.sum(
                        X[i-window_size:i]) / window_size
                else:
                    moving_average[i-1] = np.sum(X[:i]) / i
            elif type == "exp":
                if i > window_size:
                    moving_average[i-1] = np.sum(
                        np.exp(X[i-window_size:i])) / window_size
                else:
                    moving_average[i-1] = np.sum(np.exp(X[:i])) / i

        return moving_average

    def H(self, X, num):
        n = X.shape[0]
        H = np.zeros(n)

        for i in range(1, n):
            if i > num:
                H[i-1] = np.max(X[i-num:i])
                # print('H[i-1]: ', H[i-1])
            else:
                H[i-1] = np.max(X[:i])
                # print('H[i-1]: ', H[i-1])
        return H[1:]

    def L(self, X, num):
        n = X.shape[0]
        L = np.zeros(n)

        for i in range(1, n):
            if i > num:
                L[i-1] = np.min(X[i-num:i])
            else:
                L[i-1] = np.min(X[:i])

        return L[1:]

    def feature_construct(self, df):
        # Make various features that may help the predictive algorithms
        high_price = df['High'][1:]
        low_price = df['Low'][1:]
        open_price = df['Open'][1:]
        close_price = df['Close'][1:]
        adj_close = df['Adj Close'][1:]
        volume = df['Volume'][1:]
        trade_quantity = volume * close_price

        # Long term simple moving average and exponent moving avearge
        close_SMA_long = self.moving_average(close_price, 50, "simple")
        close_EMA_long = self.moving_average(close_price, 50, "exp")

        # Short term simple moving average and exponent moving avearge
        close_SMA_short = self.moving_average(close_price, 5, "simple")
        close_EMA_short = self.moving_average(close_price, 5, "exp")

        # Two features related to percentage
        highlow_percentage = (high_price - low_price) / close_price * 100.0
        percentage_change = (close_price - open_price) / open_price * 100.0

        # KLine and DLine are common features in stock predicting, the formula of
        # calculating KLine and DLine is refered from
        # "https://www.investopedia.com/articles/technical/073001.asp"
        H5 = self.H(df['Close'], 5)
        L5 = self.L(df['Close'], 5)
        H3 = self.H(df['Close'], 3)
        L3 = self.L(df['Close'], 3)
        KLine = (close_price - L5) / (H5 - L5) * 100.0
        DLine = H3 / L3 * 100.0

        # Combined the above columns to get the feature metrix
        X_pd = pd.DataFrame({'close_SMA_long': close_SMA_long, 'close_EMA_long': close_EMA_long,
                             'close_SMA_short': close_SMA_short, 'close_EMA_short': close_EMA_short,
                             'highlow_percentage': highlow_percentage, 'percentage_change': percentage_change,
                             'open': open_price, 'high': high_price, 'low': low_price, 'close': close_price,
                             'adj_close': adj_close, 'volume': volume, 'trade_quantity': trade_quantity,
                             'KLine': KLine, 'Dline': DLine})
        X_pd = X_pd[:-1]
        features = X_pd.to_numpy()
        return features

    def label_construct(self, df):
        # labels is the residuals of two continuous days calculated from the close price column
        labels = np.zeros(np.shape(df)[0] - 1)
        for i in range(np.shape(df)[0]-1):
            labels[i] = df['Close'][i+1] - df['Close'][i]

        return labels[1:]

    # Use Elastic-Net with varying alpha and lambda
    def fit(self, X, y):
        # I have also tried grid search method, which will give the similiar result as ElasticNetCV
        # elastic = ElasticNet()
        # param_grid = {
        #     'alpha': np.linspace(0, 2, num=40),
        #     'l1_ratio': np.linspace(0, 1, num=100),
        # }
        # clf = GridSearchCV(elastic, param_grid, cv=5, n_jobs=-1)
        # regr = clf.fit(X, y)

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


############################################ Problem 2 ###################################################
df = pd.read_csv("AAPL.csv")
predictor = PricePrediction(df)
regr = predictor.fit(predictor.train_X, predictor.train_y)
print("############################################ Problem 2 ############################################")
print("best alpha: ", regr.alpha_)
print("best l1_ratio: ", regr.l1_ratio_)
print("best coefficients: ", regr.coef_)

y_pred = regr.predict(predictor.test_X)
y_true = predictor.test_y
MSE, RMSE, RSQ, MAE, MAPE, MedAE = predictor.evaluate(y_true, y_pred)
print("MSE: {}, RMSE: {}, RSQ: {}, MAE: {}, MAPE: {}, MedAE: {}".format(
    MSE, RMSE, RSQ, MAE, MAPE, MedAE))

### draw close price picture ###
X = range(predictor.test_X.shape[0])
plt.subplot(121)
plt.title("Residual")
plt.plot(X, y_pred, "x-", label="y_pred residual")
plt.plot(X, y_true, "+-", label="y_true residual")
plt.legend()

plt.subplot(122)
plt.title('Close Price')
plt.plot(range(1, predictor.test_X.shape[0]+1), y_pred +
         predictor.close_test, "x-", label="y_pred close price")
plt.plot(X, predictor.close_test, "+-", label="y_true close price")
plt.legend()

plt.show()
### draw close price picture ###

##### interpret of my results #####
## Results of Stock Price Prediction ##
# The best alpha, l1_ratio and coefficients of my ElaticNetCV fit process is shown as below:
# best alpha:  1.8974358974358974
# best l1_ratio:  0.26262626262626265
# best coefficients:  [ 0.00000000e+000 -1.06009001e-099  0.00000000e+000 -1.50688745e-100
#                       0.00000000e+000 -0.00000000e+000  0.00000000e+000  0.00000000e+000
#                       0.00000000e+000  0.00000000e+000  2.67392568e-003  3.74614634e-009
#                      -1.04692329e-011 -1.10446485e-003  0.00000000e+000]

## The result from different evaluation metrics are shown as below ##
# MSE:  9.718363194530022
# RMSE:  3.1174289397723283
# RSQ:  -0.01817632468225816
# MAE:  2.3943709500293995
# MAPE:  139.96212606590632
# MedAE:  1.9408383795179005

## Features and Labels Explanation ##
# I constructed a 15-dimention feature matrix from the original data, which are the long term(50 days) simple
# moving average, long term(50 days) exponent moving avearge, short term simple moving average(5 days), short term
# exponent moving average(5 days), highlow_percentage, percentage_change over a day, the total trade quantity
# (volume * close_price) of a day, KLine, DLine. Some of these features are what I found useful in stock price
# prediction(Like KLine and Dline and some percentage related features) from the internet reference.
# For the predicting labels, I used the residual, which is the difference between today and yesterday.

## Hyper-parameters choosing ##
# I using the ElasticNetCV to perform KFold Cross Validation to choose the best value of alpha and l1_ratio.
# I have also tried the Grid search(which is shown in the comment in fit function), it gives similiar result
# as ElasticNetCV while taking longer time.

## Result Interpret ##
# I think that the stock price is related to too many factors. So, it is hard to perfectly predict tomorrow’s
# closing price using these features created from the past prices and volumes, with a simple linear model.
# From the picture I drawn, we can observe that the predicted tomorrow's close price is very near to the
# close price today. And the predicted residuals(the difference between two continuous days) is relative
# stable compared with the true difference. These indicated that my simple linear model can only give some
# tendency information, while the MSE, MAPE, RMSE are still kind of high to give the exact price prediction.
############################################ Problem 2 ###################################################


############################################ Problem 1 ###################################################
data_breast_cancer = load_breast_cancer()
X = data_breast_cancer.data
y = data_breast_cancer.target

# The new logistic regression for brier score
logistBrier = LogisticBrier()
loss_lst = logistBrier.fit(X, y)
intercept, coef = logistBrier.get_value()
print("############################################  Problem 1 ###################################################")
print("logistBrier coef: ", coef)
print("logisBrier intercept: ", intercept)
print("logisBrier loss list: ", loss_lst)

# Compare the coefficients from the regular Logistic regression and interpret the differences
logistic = LogisticRegression(
    penalty="none", solver='newton-cg', fit_intercept=False, n_jobs=-1).fit(X, y)
print("LogisticRegression coef: ", logistic.coef_)

loss_LogisticRegression = np.sum(
    (y.reshape((X.shape[0], 1)) - expit(np.dot(X, logistic.coef_.T)))**2)
print("loss of LogisticRegression: ", loss_LogisticRegression)

### interpret of my results ###
## Result Comparation ##
# 1. The result from my LogisticBrier fit function:
#    logistBrier coef:  [ 3.50382941e+01  4.49423754e-01 -2.24547118e+00 -1.78560857e-01
#                     -1.06247824e+03  1.40430451e+02 -6.02168863e+02  1.71049597e+02
#                      2.91263221e+02  4.43047458e+02  1.00140190e+02  3.63525224e+01
#                     -1.60049183e+01 -1.06068905e+00 -6.49527556e+03 -2.19758987e+03
#                      1.25054676e+03 -2.03945817e+03  2.50430349e+03  1.72586685e+04
#                     -9.89385380e+00 -4.29202969e+00  4.57249973e+00 -2.81888396e-01
#                      6.05506512e+02  3.07918645e+02 -1.22699799e+02 -3.91046236e+02
#                     -3.98174491e+02 -1.35488795e+03]
#    logisBrier intercept:  -0.0035807746074886238
#    logisBrier loss list:  [32.75658345860106, 20.80873152708887, 14.448824552947823, 10.846342156162581,
#                         7.942647114745675, 5.92221341428526, 4.461489431583983, 3.155632090958523,
#                         2.413311800764114, 2.136076018369324, 2.0433378383826764, 2.0111078246988776,
#                         1.9995836002823604, 1.9953717618145639, 1.9938068846696657, 1.9932070302615892,
#                         1.9931159729471313, 1.9929310616013955]
#
# 2. The result from LogisticRegression in sklearn.linear_model:
#    LogisticRegression coef:  [[ 7.08828763e+00 -2.00379039e-01  1.11811490e-01 -6.53888799e-02
#                             -2.15033824e+01  3.51860798e+01 -3.83327384e+01 -7.38586306e+01
#                              1.87976268e+01  1.16290172e+01 -5.94143942e+00  1.69370830e+00
#                              1.55360431e+00 -2.88224894e-01 -1.71484239e+01  4.52667015e+01
#                              6.54378234e+01 -1.93148950e+01  2.86994668e+01  9.45029025e+00
#                             -8.77315378e-01 -4.17003093e-01 -1.22858030e-01 -5.90328132e-03
#                             -3.11576096e+01  5.17111195e+00 -8.44573453e+00 -6.77641308e+01
#                             -2.52422435e+01  2.69722579e+00]]
#    loss of LogisticRegression:  6.4110720712845435

## Implement Methods ##
# I used an iterated process to calculate self.intercept and self.coef. My strategy is first fix self.beta as 0,
# then calculate self.coef. In the next iteration, I fixed self.coef and calculate self.intercept and so on...
# In each iteration, I used Newton’s method to update self.coef, which is based on the first and second derivative
# of the loss function.

## Result Interpret ##
# From the logisticBrier loss list shown above, we can see that in the training iteration process of Newton's Method,
# the loss value of logisticBrier started at 32.756583, then continued decreasing util converging.
#
# While when using the LogisticRegression of sklearn, it shows a warning that "ConvergenceWarning: newton-cg failed
# to converge. Increase the number of iterations." And the loss we get using LogisticRegression (under parameter setting:
# penalty="none", solver='newton-cg', fit_intercept=False) of sklearn is 6.4110720712845435, which is worse than the
# result we get using logisticBrier.
#
# This might because that the training process of logitsBrier is overfitted.
#
# Besides, since the Newton method I used is different from sklearn(sklearn using newton conjugategradient), this can
# explain the difference of the coef we get using sklearn LogisticRegression and logisticBrier.
#
# Since the range of expit is (0,1) and the target y's range is also (0,1), the intercept beta must be a constant closed
# to zero, which is conformed with my intercept result: logisBrier intercept:  -0.0035807746074886238.
############################################ Problem 1 ###################################################


# !!!Note!!!: I exchange the test order of Problem 1 and Problem 2 to make clear the output result, since there
# are continuous convergenceWarning logging information in the ElasticNet fit process. Otherwise the warning will
# keep the output of Problem 1 out of terminal window.
