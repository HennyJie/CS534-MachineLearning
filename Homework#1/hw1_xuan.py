'''
@Description: In User Settings Edit
@Author: your name
@Date: 2019-10-07 12:16:46
@LastEditTime: 2019-10-07 12:16:46
@LastEditors: your name
'''
# Please do not use other libraries except for numpy
import numpy as np


class Ridge:

    def __init__(self):

        self.intercept = 0
        self.coef = None

    def fit(self, X, y, coef_prior=None, lmbd=1.0):
        """Fit Ridge regression model
        Arguments:
            X {np.ndarray} -- Features
            y {np.ndarray} -- Targets
        Keyword Arguments:
            coef_prior {np.ndarray} -- A certain set of coefficients estimated by experts (default: {None})
            lmbd {float} -- Regularization strength; must be a positive float. (default: {1.0})
            !!! When "lmbd" equal to "alpha" in sklearn ridge class,
            !!! ridge(sklearn) and ridge(implement by author) can 
            !!! generate same intercept and coefficient.
        Returns:
            [int] -- 0
        """

        n, m = X.shape

        # Modify based on sklearn ridge implementation to make sure
        # ridge(sklearn) and ridge(implement by author) can generate
        # same intercept and coefficient.
        lmbd *= n

        self.coef = np.zeros(m)
        if coef_prior is None:
            coef_prior = np.zeros(m)

        # a) normalize X, x = (x - x_mu)/x_sigma
        x_mu = X.mean(axis=0)
        x_sigma = X.std(axis=0)
        X = (X-x_mu)/x_sigma

        # b) adjust coef_prior according to the normalization parameters
        # beta_prior = beta_prior * x_sigma
        coef_prior = coef_prior * x_sigma

        # c) get coefficients
        # center y, y_center = y - y_mu
        center_zero_y = y - np.mean(y)
        # beta = (X_T*X + lmbd*I)^(-1) * (lmbd*beta_prior + X_T*y_center)
        self.coef = np.dot(np.linalg.inv(np.dot(X.T, X)+lmbd*np.identity(m)),
                           (lmbd*coef_prior+np.dot(X.T, center_zero_y)))

        # d) adjust coefficients for de-normalized X
        # beta_dn = beta/x_sigma
        self.coef /= x_sigma
        # intercept = y_mu - dot_product(beta_dn, x_mu)
        self.intercept = np.mean(y) - np.sum(np.multiply(self.coef, x_mu))

        return 0

    def get_coef(self):
        """Returns the intercept and coefficients
        Returns:
            [np.float, np.ndarray] -- Weight vector and independent term in decision function
        """
        return self.intercept, self.coef


class ForwardStagewise:

    def __init__(self):
        self.intercept = 0
        self.path = []

    def fit(self, X, y, cannot_link=[], epsilon=1e-5, max_iter=1000):
        """[summary]
        Arguments:
            X {np.ndarray} -- Features
            y {np.ndarray} -- Targets
        Keyword Arguments:
            cannot_link {list} -- Model constraint (default: {[]})
            epsilon {float} -- The hyperparameter of the model (default: {1e-5})
            max_iter {int} -- Maximum number of iterations (default: {1000})
        Returns:
            [int] -- 0
        """

        # a) normalize X, x = (x - x_mu)/x_sigma
        x_mu = X.mean(axis=0)
        x_sigma = X.std(axis=0)
        X = (X-x_mu)/x_sigma

        # b-1) implement incremental forwward-stagewise
        # b-2) implement cannot-link constraints
        _, m = X.shape

        # coefficients for normalized X
        beta = np.zeros(m)
        # center y
        y_center = y - np.mean(y)
        path = [np.zeros(m)]
        constraint_links = cannot_link.copy()
        # columns can be selected during fitting process
        selected_columns = set(range(m))

        for _ in range(max_iter):
            r = y_center - np.dot(X, beta)
            mse_min, j_best, gamma_best = np.inf, 0, 0
            mses = np.zeros((m))
            for j in selected_columns:
                # Use Least Squares Method to find the jth column's coefficient
                gamma_j = np.dot(X[:, j], r)/np.dot(X[:, j], X[:, j])
                # Use mean-square error to evaluate the jth column
                mse = np.mean(np.square(r - gamma_j * X[:, j]))
                mses[j] = mse
                # find a column which can make mse least
                if mse < mse_min:
                    mse_min, j_best, gamma_best = mse, j, gamma_j
            # if the selected column in cannot_link, find the list which includes the column's index
            links = list(filter(lambda x: j_best in x, constraint_links))
            # remove other element in this list from selected_columns
            if len(links) != 0:
                constraint_links.remove(links[0])
                links[0].remove(j_best)
                selected_columns -= set(links[0])
            # Update coefficients for normalized X
            if np.abs(gamma_best) > 0:
                beta[j_best] += gamma_best * epsilon

            # c) adjust coefficients for de-normalized X
            # beta_dn = beta/x_sigma
            coef = beta / x_sigma
            # intercept = y_mu - dot_product(beta_dn, x_mu)
            self.intercept = np.mean(y) - np.sum(np.multiply(coef, x_mu))
            path.append(coef)

        # d) construct the "path" numpy array
        #     path: l+1-by-m array,
        #               where l is the total number of iterations
        #               m is the number of features in X.
        #               The first row, path[0,:], should be all zeros.
        self.path = np.array(path)

        return 0

    def get_coef_path(self):
        """Returns the intercept and coefficient
        Returns:
            [np.float, np.ndarray] -- Weight vector and independent term in decision function
        """
        return self.intercept, self.path
