'''
@Description: Class Ridge and ForwardStagewise, CS534-Machine Learning Hw1 
@Author: Hejie Cui
@Date: 2019-09-19 10:54:23
@LastEditTime: 2019-09-23 18:04:34
'''
# Please do not use other libraries except for numpy

import numpy as np


class Ridge:
    """Class for Ridge Regression with Prior Coefficients."""

    def __init__(self):
        self.intercept = 0
        self.coef = None

    def fit(self, X, y, coef_prior=None, lmbd=1.0):
        """
        Parameters
        ----------
        X : ndarray
            The input data
        y : ndarray
            The target data
        coef_prior : ndarray, optional
            Prior knowledge, which will be reflected in the penalty term of the loss function
        lmbd : float, optional
            Regularization strength, which has the same meaning as the alpha in SKlearn regression but with 
            a scale difference. That means we need to multiply the lmbd with the sample numbers in the fit 
            function, that is n, if we want to compare with passing the same value of alpha to the ridge 
            regression of sklearn.
            For example, if we passed the lmbd=1.0 to this fit function and alpha=1.0 to the ridge regression 
            of sklearn, we need to multiply the lmbd with the sample numbers of X (That is n) in the fit function 
            before figuring out the closed form solution.
            By doing so, I can get the exactly the same values of intercept and self.coef values compared with sklearn.
        """

        n, m = X.shape
        self.coef = np.zeros(m)
        if coef_prior.any() == None:
            coef_prior = np.zeros(m)

        # a) normalize X
        x_mu = np.average(X, axis=0)
        x_sigma = np.std(X, axis=0)
        for i in range(m):
            X[:, i] = (X[:, i] - x_mu[i]) / x_sigma[i]

        # b) adjust coef_prior according to the normalization parameters
        for i in range(m):
            coef_prior[i] = coef_prior[i] * x_sigma[i]

        # c) get coefficients
        # First, we transform y to have 0 mean. If data are first centered about 0, then favoring small
        # intercept not so worrisome. Using the loss function, we can get the closed form solution, which
        # is shown in the calculation formula of self.coef
        I = np.eye(m)
        centered_y = y - np.mean(y)
        # the reason why we need to multiply lmbd with n is explained in the comments of fit function's
        lmbd *= n
        self.coef = np.dot(np.linalg.inv(np.dot(X.T, X) + lmbd*I),
                           (lmbd * coef_prior + np.dot(X.T, centered_y)))

        # d) adjust coefficients for de-normalized X
        self.intercept = np.mean(y)
        for i in range(m):
            self.intercept -= self.coef[i] / x_sigma[i] * x_mu[i]
            self.coef[i] = self.coef[i] / x_sigma[i]

        return 0

    def get_coef(self):
        return self.intercept, self.coef


class ForwardStagewise:
    """Class for Incremental Forward-Stagewise with Cannot-Link Constraints."""

    def __init__(self):
        self.intercept = 0
        self.path = []

    def fit(self, X, y, cannot_link=[], epsilon=1e-2, max_iter=1000):
        """
        Parameters
        ----------
        X : ndarray
            The input data
        y : ndarray
            The target data
        cannot_link : list of list, optional
            The groups in which at most one of the features in the same group should be active
        epsilon : float, optional
            The updated coefficient
        max_iter: int, optional
            The maximum interation number
        """

        # a) normalize X
        n, m = X.shape
        x_mu = np.average(X, axis=0)
        x_sigma = np.std(X, axis=0)
        for i in range(m):
            X[:, i] = (X[:, i] - x_mu[i]) / x_sigma[i]

        # b-1) implement incremental forwward-stagewise
        # b-2) implement cannot-link constraints
        nsteps = 1000
        beta = np.zeros(m)
        y_copy = y
        y = y - np.mean(y)

        # a dictionary recording whether the key column is still be updating
        column_is_updating = {k: v for (k, v) in zip(range(m), np.ones(m))}
        path = []
        path.append(np.zeros(m))

        for s in range(nsteps):
            r = y - np.dot(X, beta)
            mse_min, j_best, gamma_best = np.inf, 0, 0
            # filter the updating columns
            updating_columns = [i for i, is_updating in column_is_updating.items()
                                if is_updating == 1]
            # use mse to evaluate which is the best column to choose
            for j in updating_columns:
                gamma_j = np.dot(X[:, j], r) / np.dot(X[:, j], X[:, j])
                mse = np.mean(np.square(r - gamma_j * X[:, j]))
                if mse < mse_min:
                    mse_min, j_best, gamma_best = mse, j, gamma_j

            # find the group index in the cannot_link which the j_best belong to
            group_index = [i for i, group_members in enumerate(cannot_link)
                           if j_best in group_members]

            # if the j_best belongs to a group in the cannot_link, then all the other columns
            # in this group will no longer be updated
            if len(group_index):
                for i in cannot_link[group_index[0]]:
                    column_is_updating[i] = 0
                column_is_updating[j_best] = 1
                cannot_link.remove(cannot_link[group_index[0]])

            if np.abs(gamma_best) > 0:
                beta[j_best] += gamma_best * epsilon
                # append the beta for denomalized X to path list
                path.append(beta.tolist())

        # c) adjust coefficients for de-normalized X
        self.intercept = np.mean(y_copy)
        for i in range(m):
            self.intercept -= beta[i] / x_sigma[i] * x_mu[i]
            beta[i] = beta[i] / x_sigma[i]

        # d) construct the "path" numpy array
        #     path: (l+1)-by-m array,
        #               where l is the total number of iterations
        #               m is the number of features in X.
        #               The first row, path[0,:], should be all zeros.

        # transform the path list into numpy array and adjust the beta values in path for the denormalized X
        self.path = np.array(path)
        for j in range(m):
            self.path[:, j] = self.path[:, j] / x_sigma[j]

        return 0

    def get_coef_path(self):
        # return the intercept for denormalized X and each line in path is fitted for the denormalized X
        return self.intercept, self.path
