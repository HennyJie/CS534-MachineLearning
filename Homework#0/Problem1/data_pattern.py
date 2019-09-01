import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.linalg
import scipy.optimize as optimize
from scipy.optimize import curve_fit
import numpy as np
from matplotlib import cm
import pandas_profiling

data = pd.read_csv("Homework#0/Problem1/hw0_p1.csv")

# generate a little basic for serious exploratory data analysis using pandas_profiling
profile = data.profile_report(title='Pandas Profiling Report')
profile.to_file(output_file="output.html")

data = np.array(data.values.tolist())
x1 = data[:,0]
x2 = data[:,1]
y = data[:,2]

# best-fit quadratic surface
A = np.c_[np.ones(data.shape[0]), data[:,:2], np.prod(data[:,:2], axis=1), data[:,:2]**2]
C, _, _, _ = scipy.linalg.lstsq(A, y)
print("Estimated Surface Function: y=({})+({}*x1)+({}*x2)+({}*x1x2)+({}*x1^2)+({}*x2^2)".format(C[0], C[1], C[2], C[3], C[4], C[5]))

# regular grid covering the domain of the data
X, Y = np.meshgrid(np.arange(x1.min(), x1.max(), 0.05), np.arange(x2.min(), x2.max(), 0.05))
XX = X.flatten()
YY = Y.flatten()
Z = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX*YY, XX**2, YY**2], C).reshape(X.shape)

# draw surface
ax = plt.figure().gca(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.2)
ax.scatter(x1, x2, y)
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')
plt.savefig("Homework#0/Problem1/data_pattern.png")

# calculate the correlation between estimated curve and original scatter points
pre_y = np.dot(np.c_[np.ones(data.shape[0]), x1, x2, x1*x2, x1**2, x2**2], C)
data = pd.read_csv("Homework#0/Problem1/hw0_p1.csv")
print("The correlation between the surface fitting model using the least squares method and y is: ", data['y'].corr(pd.Series(data=pre_y)))
