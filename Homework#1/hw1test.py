'''
@Description: In User Settings Edit
@Author: your name
@Date: 2019-10-07 11:44:34
@LastEditTime: 2019-10-07 16:03:59
@LastEditors: Please set LastEditors
'''
import numpy as np
from hw1 import ForwardStagewise
from sklearn.linear_model import Ridge as skRidge
from sklearn import datasets


def test_p2(X, y, cannot_link):

    model = ForwardStagewise()
    print(cannot_link)
    model.fit(X, y, cannot_link)
    # model.fit(X, y, cannot_link.copy())
    intercept, path = model.get_coef_path()
    print(cannot_link)

    coef = path[-1, :]  # last path
    # print("coef of cannot_link {}: {}".format(cannot_link, coef))

    y_hat_fs = intercept + np.dot(X, coef)
    mse_fs = np.mean(np.square(y - y_hat_fs))

    cannot_stat = 1
    for group in cannot_link:
        cannot_stat = cannot_stat * np.prod([coef[j] for j in group])

    # print("cannot_stat: ", cannot_stat)
    # print("cannot_stat of kan: ", cannot_stat)

    # regular linear regression
    intercept_lm = np.mean(y)
    z = y - intercept_lm
    coef_lm = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, z))
    y_hat_lm = intercept_lm + np.dot(X, coef_lm)
    mse_lm = np.mean(np.square(y - y_hat_lm))

    mse_diff = np.abs(mse_fs - mse_lm)

    test_a = mse_diff < 100.0
    test_b = cannot_stat == 0

    return test_a, test_b


if __name__ == "__main__":

    data = datasets.load_boston()
    X = data["data"]
    y = data["target"]
    n, m = X.shape

    # # Problem 2
    print(test_p2(X, y, []))  # TF
    print(test_p2(X, y, [[3, 4]]))  # TT
    print(test_p2(X, y, [[3, 4], [1, 2]]))  # FT
