# Please do not use other libraries except for numpy
import numpy as np
from sklearn.datasets import load_boston
from sklearn.linear_model import Ridge


class Ridge:

    def __init__(self):
        self.intercept = 0
        self.coef = None

    def loss_func(self, y, X, coef, coef_prior, lmbd, intercept):
        return(np.sum(np.square(y - intercept - np.dot(X, coef))) + lmbd * np.sum(np.square(coef - coef_prior)))

    def fit(self, X, y, coef_prior=None, lmbd=1.0):
        n, m = X.shape
        self.coef = np.zeros(m)
        if coef_prior.any() == None:
            coef_prior = np.zeros(m)

        X = np.c_[np.ones(n), X]
        coef_prior = np.insert(coef_prior, 0, 0, axis=0)

        # a) normalize X
        x_mu = np.average(X, axis=0)
        x_sigma = np.std(X, axis=0)
        for i in range(1, m + 1):
            X[:, i] = (X[:, i] - x_mu[i]) / x_sigma[i]
        # print("X after normalize: ", X)

        # b) adjust coef_prior according to the normalization parameters
        for i in range(m):
            coef_prior[i] = coef_prior[i] * x_sigma[i]

        # c) get coefficients
        I = np.eye(m+1)
        I[0][0] = 0
        self.coef = np.dot(np.linalg.inv(np.dot(X.T, X) + lmbd*I),
                           (lmbd * coef_prior + np.dot(X.T, y)))
        self.intercept = self.coef[0]
        print("self.intercept before denormalize: ", self.intercept)
        print("self.coef before denormalize: ", self.coef[1:])

        # print("self.coef before normalized: ", self.coef)
        # print("self.intercept before normalized: ", self.intercept)
        # current_coef = np.zeros(m)
        # current_intercept = np.zeros(n)
        # threshold_coef = 1e-4
        # threshold_intercept = 1e-4
        # coef_difference = np.inf
        # intercept_difference = np.inf

        # while coef_difference > threshold_coef:
        #     new_intercept = y - np.dot(X, current_coef)
        #     intercept_difference = np.mean(
        #         np.abs(new_intercept - current_intercept))

        #     XTX = np.dot(X.T, X)
        #     lmbdI = lmbd * np.eye(m)
        #     lmbdCoef_prior = lmbd * coef_prior
        #     XTY = np.dot(X.T, y)
        #     XTIntercept = np.dot(X.T, new_intercept)

        #     new_coef = np.dot(np.linalg.inv(XTX + lmbdI),
        #                       (lmbdCoef_prior + XTY - XTIntercept))
        #     coef_difference = np.mean(np.abs(new_coef - current_coef))

        #     current_intercept = new_intercept
        #     current_coef = new_coef

        # self.intercept = current_intercept
        # self.coef = current_coef

        # d) adjust coefficients for de-normalized X
        for i in range(1, m+1):
            self.intercept -= self.coef[i] / x_sigma[i] * x_mu[i]
            self.coef[i] = self.coef[i] / x_sigma[i]

        data = load_boston()
        X = data.data
        y = data.target
        self.coef = self.coef[1:]
        coef_prior = coef_prior[1:]
        # X = X[:, 1:]
        loss = self.loss_func(y, X, self.coef, coef_prior,
                              lmbd, self.intercept)
        loss /= n
        print("loss: ", loss)

        return 0

    def get_coef(self):
        return self.intercept, self.coef


class ForwardStagewise:

    def __init__(self):
        self.intercept = 0
        self.path = []

    def fit(self, X, y, cannot_link=[], epsilon=1e-2, max_iter=1000):

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
        y = y - np.mean(y)

        column_is_updating = {k: v for (k, v) in zip(range(m), np.ones(m))}
        path = []
        path.append(np.zeros(m))

        for s in range(nsteps):
            r = y - np.dot(X, beta)
            mse_min, j_best, gamma_best = np.inf, 0, 0
            updating_columns = [i for i, is_updating in column_is_updating.items()
                                if is_updating == 1]
            for j in updating_columns:
                gamma_j = np.dot(X[:, j], r) / np.dot(X[:, j], X[:, j])
                mse = np.mean(np.square(r - gamma_j * X[:, j]))
                if mse < mse_min:
                    mse_min, j_best, gamma_best = mse, j, gamma_j

            group_index = [i for i, group_members in enumerate(cannot_link)
                           if j_best in group_members]

            if len(group_index):
                for i in cannot_link[group_index[0]]:
                    column_is_updating[i] = 0
                column_is_updating[j_best] = 1
                cannot_link.remove(cannot_link[group_index[0]])

            if np.abs(gamma_best) > 0:
                beta[j_best] += gamma_best * epsilon
                path.append(beta.tolist())
        print("cannot link beta before denormalize: ", beta)

        # c) adjust coefficients for de-normalized X
        self.intercept = np.mean(y)
        for i in range(m):
            self.intercept -= beta[i] / x_sigma[i] * x_mu[i]
            beta[i] = beta[i] / x_sigma[i]
        print("cannot link beta after denormalize: ", beta)
        print("self.intercept after denormalize: ", self.intercept)

        # d) construct the "path" numpy array
        #     path: l-by-m array,
        #               where l is the total number of iterations
        #               m is the number of features in X.
        #               The first row, path[0,:], should be all zeros.
        path = path[:1000]
        self.path = np.array(path)

        return 0

    def get_coef_path(self):
        return self.intercept, self.path
