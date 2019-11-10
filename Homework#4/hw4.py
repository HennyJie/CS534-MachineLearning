'''
@Author: your name
@Date: 2019-11-09 19:31:31
@LastEditTime: 2019-11-09 21:46:18
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /Machine Learning/Homework#4/hw4.py
'''
import numpy as np
from sklearn.datasets import load_breast_cancer


class GreedyKNN:
    def get_feature_order(X, y, k=5):

        n, m = X.shape
        feature_lst = []

        while len(feature_lst) < m:
            max_auroc = 0.0
            max_var = -1
            for j in range(m):
                if j in feature_lst:
                    continue
                # Implement your own kNNpredict function
                # The function should return kNN prediction for the given X, y, and k
                y_hat = kNNpredict(X[, feature_lst + [j]], y, k)  # TODO
                auroc =  # TODO: measure AUROC with y_hat and y
                if auroc > max_auroc:
                    max_auroc = auroc
                    max_var = j
            feature_lst.append(max_var)

        return feature_lst
