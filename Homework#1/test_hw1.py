import numpy as np
from sklearn.datasets import load_boston
from sklearn import linear_model
from hw1 import Ridge
from hw1 import ForwardStagewise
from sklearn.preprocessing import scale

######################## Test Ridge Regression #########################
# data = load_boston()
# X = data.data
# y = data.target
# n, m = X.shape
# print("n: {}, m: {}".format(n, m))
# coef_prior = np.zeros(m)
# ridge = Ridge()
# ridge.fit(X, y, coef_prior=coef_prior, lmbd=1)
# intercept, coef = ridge.get_coef()
# print("intercept after denormalize:: {}, coef:{} ".format(intercept, coef))

# data = load_boston()
# X = data.data
# y = data.target
# # reg_sk = linear_model.Ridge(alpha=1, normalize=True, solver='cholesky')
# reg_sk = linear_model.Ridge(alpha=1, solver='cholesky')
# reg_sk.fit(X, y)
# intercept_sk = reg_sk.intercept_
# coef_sk = reg_sk.coef_
# print("intercept_sk:{}, coef_sk:{}".format(intercept_sk, coef_sk))
##########################################################################

########################### Test Stagewise Regression ####################
data = load_boston()
X = data.data
n, m = X.shape
y = data.target
cannot_link = [[0, 1], [2, 3, 4], [5, 6]]
fsw = ForwardStagewise()
fsw.fit(X, y, cannot_link=cannot_link)
intercept, path = fsw.get_coef_path()
# print("cannot link intercept:", intercept)
# print("cannot link path:", path)

# # Forward Stagewise
# data = load_boston()
# X = scale(data.data)
# n, m = X.shape
# y = data.target
# columns = data.feature_names.tolist()

# nsteps = 1000
# delta = 1e-2
# beta = np.zeros(m)
# y = y - np.mean(y)

# path = []
# for s in range(nsteps):
#     r = y - np.dot(X, beta)
#     mse_min, j_best, gamma_best = np.inf, 0, 0
#     for j in range(m):
#         gamma_j = np.dot(X[:, j], r)/np.dot(X[:, j], X[:, j])
#         mse = np.mean(np.square(r - gamma_j * X[:, j]))
#         if mse < mse_min:
#             mse_min, j_best, gamma_best = mse, j, gamma_j
#     if np.abs(gamma_best) > 0:
#         beta[j_best] += gamma_best * delta
#         path.append(beta.tolist())

# print("forward stagewise beta", beta)
# print("forward stagewise path: ", path)
##########################################################################
