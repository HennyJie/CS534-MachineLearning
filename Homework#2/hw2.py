'''
@Description: Logistic Regression for Brier Score and Price Predicting, CS534-Machine Learning Hw2
@Author: Hejie Cui
@Date: 2019-09-28 18:35:20
@LastEditTime: 2019-09-28 21:17:40
'''
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import ElasticNet
import pandas as pd

'''
@description: Problem 1
@param {type} 
@return: 
'''
data_breast_cancer = load_breast_cancer()
print("data_breast_cancer.data shape: ", np.shape(data_breast_cancer.data))
print("data_breast_cancer.target shape: ", np.shape(data_breast_cancer.target))


'''
@description: Problem 2
@param {type} 
@return: 
'''
data_apple = pd.read_csv("AAPL.csv")
