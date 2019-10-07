'''
@Description: In User Settings Edit
@Author: your name
@Date: 2019-10-07 16:19:31
@LastEditTime: 2019-10-07 16:49:55
@LastEditors: Please set LastEditors
'''
import numpy as np
n = 5
m = 3
orig_X = np.random.random_sample(size=(m, m))
print("orig_X", orig_X)

X1 = orig_X
print("X1", X1)
X2 = orig_X
print("X2", X2)

x_mu = np.average(X1, axis=0)
x_sigma = np.std(X1, axis=0)

# X = orig_X
# your normalization
# print("original X: ", X)

for i in range(m):
    # print("i: ", i)
    # print("X[:, i] before", X[:, i])
    # print("x_mu[i]: ", x_mu[i])
    # print("x_sigma[i]", x_sigma[i])

    X1[:, i] = (X1[:, i] - x_mu[i]) / x_sigma[i]
    # print("X[:, i]", X[:, i])
print("result of normalization using your script")
print(X1)

# X = orig_X
# # print("original X2: ", X)
X2 = (X2 - x_mu) / x_sigma
print("correct normalization")
print(X2)
