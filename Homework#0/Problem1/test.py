import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
from mpl_toolkits.mplot3d import Axes3D


def fun(x):
    if x >= 0:
        return '+'+str(x)
    else:
        return str(x)


data = pd.read_csv("Homework#0/Problem1/hw0_p1.csv")

data = np.array(data.values.tolist())

x1 = data[:,0]
x2 = data[:,1]
y = data[:,2]

# best-fit quadratic curve
A = np.c_[np.ones(data.shape[0]), data[:,:2], np.prod(data[:,:2], axis=1), data[:,:2]**2]
print("A: ", A)
C, _, _, _ = scipy.linalg.lstsq(A, y)

# regular grid covering the domain of the data
X, Y = np.meshgrid(np.arange(x1.min(), x1.max(), 0.05), np.arange(x2.min(), x2.max(), 0.05))
XX = X.flatten()
YY = Y.flatten()

Z = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX*YY, XX**2, YY**2], C).reshape(X.shape)
print("y={}*x1^2{}*x1x2{}*x2^2{}*x1{}*x2{}".format(fun(C[0]), fun(C[1]), fun(C[2]), fun(C[3]), fun(C[4]), fun(C[5])))

# plot points and fitted surface
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.2)
ax.scatter(data[:,0], data[:,1], data[:,2], c='r', s=50)
plt.xlabel('x1')
plt.ylabel('x2')
ax.set_zlabel('y')
ax.axis('equal')
ax.axis('tight')
# plt.show()

pre_y = np.dot(np.c_[np.ones(x1.shape), x1, x2, x1*x2, x1**2, x2**2], C)

data = pd.read_csv("Homework#0/Problem1/hw0_p1.csv")
print(data['y'].corr(pd.Series(data=pre_y)))
