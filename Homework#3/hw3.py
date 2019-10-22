'''
@Description: Machine Learning Hw3
@Author: Hejie Cui
@Date: 2019-10-08 15:46:41
@LastEditTime: 2019-10-21 22:02:07
'''
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_boston
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Ridge
from math import log


class QuarternaryDecisionTree:
    def __init__(self, max_depth=2):
        self.max_depth = max_depth
        self.tree = None

    # Calculate the entropy(entropy = -info_gain) for a specific node
    def get_entropy(self, target):
        if target.size != 0:
            label = np.unique(target)
            n = label.size
            count = np.zeros(n)
            p_i = np.zeros(n)

            # since there are only 0 and 1, p[i]=count/target.size is equal to the average value of node[i]
            for i in range(n):
                count[i] = target[target == label[i]].size
            p_i = np.divide(count, target.size)

            # the entropy(-info_gain) of each node
            entropy = 0
            for i in range(n):
                entropy = entropy - p_i[i] * np.log2(p_i[i])
        else:
            entropy = 0

        return entropy

    # Calculate the entropy under the condition that two given feature and two given values
    def get_condition_entropy(self, feature1, value1, feature2, value2, target):
        N1_index = np.logical_and(feature1 <= value1, feature2 <= value2)
        N2_index = np.logical_and(feature1 <= value1, feature2 > value2)
        N3_index = np.logical_and(feature1 > value1, feature2 <= value2)
        N4_index = np.logical_and(feature1 > value1, feature2 > value2)

        target_N1 = target[N1_index]
        target_N2 = target[N2_index]
        target_N3 = target[N3_index]
        target_N4 = target[N4_index]

        # the entropy under the condition that two given feature and two given values is
        # the entropy sum of four nodes
        entropy = target_N1.size * self.get_entropy(target_N1) + \
            target_N2.size * self.get_entropy(target_N2) + \
            target_N3.size * self.get_entropy(target_N3) + \
            target_N4.size * self.get_entropy(target_N4)

        return entropy

    # Generate all the possible split values for a specifit feature
    # First, I sorted the feature based on its value(from small to big), then I set a step as the size of unique values//10
    # which means I will divide then feature values into ten folder and select the border value of each folder.
    def generate_split_values(self, feature, target):
        argsort = feature.argsort()
        f1 = feature[argsort]
        values = np.unique(f1)
        step = values.size//10
        split_values = []

        # We add one value to the split values list per step size
        for i in range(1, 10):
            index = i*step
            split_values.append(values[index])

        return np.array(split_values)

    # Calculate the minimal entropy(the max info_gain) for two specific features
    def get_features_pair_min_entropy(self, feature1, feature2, target):
        min_entropy = float('inf')
        min_S1 = 0
        min_S2 = 0
        values1 = self.generate_split_values(feature1, target)
        values2 = self.generate_split_values(feature2, target)

        for v1 in values1:
            for v2 in values2:
                entropy = self.get_condition_entropy(
                    feature1, v1, feature2, v2, target)
                # Here I try to find the minimal entropy, which is equals to find the maximax info_gain
                if entropy < min_entropy:
                    min_entropy = entropy
                    min_S1 = v1
                    min_S2 = v2

        return min_S1, min_S2, min_entropy

    # Find the best two split pairs with the max info_gain (min entropy)
    def select_split_pairs(self, X, y):
        max_info_gain = -float('inf')
        features_num = X.shape[1]
        selected_X1 = None
        selected_X2 = None
        selected_S1 = 0
        selected_S2 = 0

        for i in range(features_num):
            for j in range(features_num):
                if i == j:
                    continue
                else:
                    S1, S2, entropy = self.get_features_pair_min_entropy(
                        X[:, i], X[:, j], y)
                    info_gain = -entropy
                    if info_gain > max_info_gain:
                        max_info_gain = info_gain
                        selected_X1 = i
                        selected_S1 = S1
                        selected_X2 = j
                        selected_S2 = S2

        return selected_X1, selected_S1, selected_X2, selected_S2

    def fit(self, X, y):
        self.tree = self.decision_tree(X, y, self.max_depth, np.mean(y))

    # The quarternary decision tree construction logic
    def decision_tree(self, X, y, max_depth, parent_node_value):
        n, m = X.shape
        # if a node is empty(0 samples in this node), then we use the predicting value
        # of its parent node as the output of current node.
        if n < 3 or max_depth == 0:
            if n == 0:
                return parent_node_value
            return np.mean(y)

        X1, S1, X2, S2 = self.select_split_pairs(X, y)

        N1_index = np.logical_and(X[:, X1] <= S1, X[:, X2] <= S2)
        N2_index = np.logical_and(X[:, X1] <= S1, X[:, X2] > S2)
        N3_index = np.logical_and(X[:, X1] > S1, X[:, X2] <= S2)
        N4_index = np.logical_and(X[:, X1] > S1, X[:, X2] > S2)

        X_N1, y_N1 = X[N1_index, :], y[N1_index]
        X_N2, y_N2 = X[N2_index, :], y[N2_index]
        X_N3, y_N3 = X[N3_index, :], y[N3_index]
        X_N4, y_N4 = X[N4_index, :], y[N4_index]

        # Store the tree in a dictionay
        return {'split_var_1': X1,
                'split_value_1': S1,
                'split_var_2': X2,
                'split_value_2': S2,
                'N1': self.decision_tree(X_N1, y_N1, max_depth-1, np.mean(y)),
                'N2': self.decision_tree(X_N2, y_N2, max_depth-1, np.mean(y)),
                'N3': self.decision_tree(X_N3, y_N3, max_depth-1, np.mean(y)),
                'N4': self.decision_tree(X_N4, y_N4, max_depth-1, np.mean(y))}

    def predict(self, X):
        y_pred = []
        for x in X:
            y_pred.append(self.single_sample_predict(x, self.tree))
        return y_pred

    # Recursive single sample predict function
    def single_sample_predict(self, x, tree):
        if not isinstance(tree, dict):
            return tree

        if x[tree['split_var_1']] <= tree['split_value_1'] and x[tree['split_var_2']] <= tree['split_value_2']:
            return self.single_sample_predict(x, tree['N1'])
        elif x[tree['split_var_1']] <= tree['split_value_1'] and x[tree['split_var_2']] > tree['split_value_2']:
            return self.single_sample_predict(x, tree['N2'])
        elif x[tree['split_var_1']] > tree['split_value_1'] and x[tree['split_var_2']] <= tree['split_value_2']:
            return self.single_sample_predict(x, tree['N3'])
        elif x[tree['split_var_1']] > tree['split_value_1'] and x[tree['split_var_2']] > tree['split_value_2']:
            return self.single_sample_predict(x, tree['N4'])


class DaRDecisionTree:
    def __init__(self, max_depth=2):
        self.max_depth = max_depth
        self.tree = None

    # Calculate the MSE for a specific feature and a specific value
    def get_condition_MSE(self, feature, value, target):
        L_index = feature <= value
        R_index = feature > value

        target_L = target[L_index]
        target_R = target[R_index]

        if target_L.size == 0:
            L_MSE = 0
        else:
            L_MSE = target_L.size * np.var(target_L)
        if target_R.size == 0:
            R_MSE = 0
        else:
            R_MSE = target_R.size * np.var(target_R)
        MSE = L_MSE + R_MSE

        return MSE

    # Get the minimal MSE for a specific feature
    def get_feature_min_MSE(self, feature, target):
        min_MSE = float('inf')
        min_S = 0
        argsort = feature.argsort()
        f1 = feature[argsort]

        for v in f1:
            MSE = self.get_condition_MSE(
                feature, v, target)
            if MSE < min_MSE:
                min_MSE = MSE
                min_S = v

        return min_S, min_MSE

    # Select the best split pairs with minimal MSE
    def select_split_pairs(self, X, y):
        min_MSE = float('inf')
        features_num = X.shape[1]
        min_X1 = None
        min_X2 = None
        min_S1 = 0
        min_S2 = 0

        for i in range(features_num):
            S, MSE = self.get_feature_min_MSE(X[:, i], y)

            if MSE < min_MSE:
                min_MSE = MSE
                min_X = i
                min_S = S

        return min_X, min_S

    def fit(self, X, y):
        self.tree = self.decision_tree(X, y, self.max_depth)

    # The divide-and-regress tree construction logic
    def decision_tree(self, X, y, max_depth):
        n, m = X.shape
        # At each leaf node, apply a linear regression
        if n < 3 or max_depth == 0:
            r = Ridge()
            r.fit(X, y)
            return r

        j_best, s_value = self.select_split_pairs(X, y)

        L_index = X[:, j_best] <= s_value
        R_index = X[:, j_best] > s_value

        X_L, y_L = X[L_index, :], y[L_index]
        X_R, y_R = X[R_index, :], y[R_index]

        return {'split_var': j_best,
                'split_value': s_value,
                'L': self.decision_tree(X_L, y_L, max_depth-1),
                'R': self.decision_tree(X_R, y_R, max_depth-1)}

    def predict(self, X):
        y_pred = []
        for x in X:
            y_pred.append(self.single_sample_predict(x, self.tree))
        return y_pred

    # The recursive single sample prediction function
    def single_sample_predict(self, x, tree):
        if not isinstance(tree, dict):
            return tree.predict(x.reshape(1, -1))[0]

        if x[tree['split_var']] <= tree['split_value']:
            return self.single_sample_predict(x, tree['L'])
        elif x[tree['split_var']] > tree['split_value']:
            return self.single_sample_predict(x, tree['R'])


################################## Test Script for Problem 1 ##################################
data_breast_cancer = load_breast_cancer()
feature_names = data_breast_cancer.feature_names
class_names = data_breast_cancer.target_names

X = data_breast_cancer.data
y = data_breast_cancer.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=100)
clf = QuarternaryDecisionTree()

# - Not split the train and test set
clf.fit(X, y)
y_pred = clf.predict(X)
print("==================Problem 1====================")
print("AUC: ", metrics.roc_auc_score(y, y_pred))
print("Quarternary Decision Tree Structure: ", clf.tree)

# - Split the train and test set
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# print("==================Problem 1====================")
# print("AUC: ", metrics.roc_auc_score(y_pred, y_pred))
# print("clf.tree: ", clf.tree)

################################## Interpretation for Problem 1 ##################################
# 1. This is what I got if we don't split the train and test set, just using the original X and y to
# construct a decision tree, then use this decision tree to predict X:
# (1) AUC:  0.9558702791461412
# (2) Quarternary Decision Tree Structure:
# clf.tree:  {'split_var_1': 7, 'split_value_1': 0.04908, 'split_var_2': 23, 'split_value_2': 867.1,
# 'N1': {'split_var_1': 7, 'split_value_1': 0.02854, 'split_var_2': 15, 'split_value_2': 0.01174,
# 'N1': 1.0, 'N2': 0.9902912621359223, 'N3': 0.25, 'N4': 0.9855072463768116},
# 'N2': {'split_var_1': 0, 'split_value_1': 13.61, 'split_var_2': 1, 'split_value_2': 18.29,
# 'N1': 0.5, 'N2': 0.0, 'N3': 1.0, 'N4': 0.0},
# 'N3': {'split_var_1': 1, 'split_value_1': 20.76, 'split_var_2': 28, 'split_value_2': 0.3527,
# 'N1': 0.8846153846153846, 'N2': 0.0, 'N3': 0.0, 'N4': 0.1111111111111111},
# 'N4': {'split_var_1': 0, 'split_value_1': 15.32, 'split_var_2': 1, 'split_value_2': 17.35,
# 'N1': 0.0, 'N2': 0.0, 'N3': 0.0, 'N4': 0.0}}
#
# 2. This is what I got if spliting the train and test set of ratio 0.75:0.25, using the X_train and y_train to
# construct a decision tree, then use this decision tree to predict X_test:
# (1) AUC:  0.9952235611225622
# (2) Quarternary Decision Tree Structure:
# clf.tree:  {'split_var_1': 7, 'split_value_1': 0.04951, 'split_var_2': 23, 'split_value_2': 922.8,
# 'N1': {'split_var_1': 1, 'split_value_1': 21.26, 'split_var_2': 22, 'split_value_2': 92.0,
# 'N1': 0.994535519125683, 'N2': 1.0, 'N3': 1.0, 'N4': 0.7083333333333334},
# 'N2': {'split_var_1': 23, 'split_value_1': 947.9, 'split_var_2': 29, 'split_value_2': 0.06469,
# 'N1': 0.0, 'N2': 1.0, 'N3': 1.0, 'N4': 0.0},
# 'N3': {'split_var_1': 20, 'split_value_1': 14.91, 'split_var_2': 21, 'split_value_2': 24.85,
# 'N1': 1.0, 'N2': 0.5, 'N3': 0.7368421052631579, 'N4': 0.0},
# 'N4': {'split_var_1': 0, 'split_value_1': 15.46, 'split_var_2': 1, 'split_value_2': 17.25,
# 'N1': 0.0, 'N2': 0.0, 'N3': 0.0, 'N4': 0.0}}
#
# 3. Interpret the selected splitting pairs:
# Different from the normal decision, there are for loops for quarternary decision tree(Two for variables and two for values):
# For the features, I just loop over all possible combinations of two feature (x1, x2).
# For the values, I loop over all possible values of each combination of variables:
# My method for choosing all the possible values for a specific variable:
#   First, I sorted the feature based on its value(from small to big), then I set a step as the size of unique values//10
#   which means I will divide then feature values into ten folder and select one value every step(values//10). I didn't
#   choose the border value(the least and the largest value for each feature) because this will introduce some empty nodes
#   which will include no samples.
# And this is what I got for the first partition:
# {'split_var_1': 7, 'split_value_1': 0.04951, 'split_var_2': 23, 'split_value_2': 922.8}
# I also tried different methods of producing all possible values for a specific variables, such as setting the intervals
# bewteen the values. The results shows great difference of pair choosing result, so I guess that the choose of values
# will greatly influence the split pair selection results.

################################## Test Script for Problem 2 ##################################
data_boston = load_boston()
X = data_boston.data
y = data_boston.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=45)
clf = DaRDecisionTree()

# - Not split the train and test set
clf.fit(X, y)
y_pred = clf.predict(X)
MSE = metrics.mean_squared_error(y, y_pred)
RMSE = np.sqrt(MSE)
RSQ = metrics.r2_score(y, y_pred)
MAE = metrics.mean_absolute_error(y, y_pred)
MAPE = np.mean(np.abs((y - y_pred) / y)) * 100
MedAE = metrics.median_absolute_error(y, y_pred)
print("==================Problem 2====================")
print("MSE: {}, RMSE: {}, RSQ: {}, MAE: {}, MAPE: {}, MedAE: {}".format(
    MSE, RMSE, RSQ, MAE, MAPE, MedAE))
print("The Divide-and-Regress Tree Structure: ", clf.tree)

# - Split the train and test set
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# MSE = metrics.mean_squared_error(y_test, y_pred)
# RMSE = np.sqrt(MSE)
# RSQ = metrics.r2_score(y_test, y_pred)
# MAE = metrics.mean_absolute_error(y_test, y_pred)
# MAPE = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
# MedAE = metrics.median_absolute_error(y_test, y_pred)
# print("==================Problem 2====================")
# print("MSE: {}, RMSE: {}, RSQ: {}, MAE: {}, MAPE: {}, MedAE: {}".format(
#     MSE, RMSE, RSQ, MAE, MAPE, MedAE))
# print("clf.tree: ", clf.tree)

################################## Interpretation for Problem 2 ##################################
# 1. This is what I got if we don't split the train and test set, just using the original X and y to
# construct a decision tree, then use this decision tree to predict X:
# (1) Different Evaluation Methods:
# MSE: 10.471475481801459, RMSE: 3.2359659271694223, RSQ: 0.8759591265508367,
# MAE: 2.268694453989851, MAPE: 11.629639402505596, MedAE: 1.7526924117532339
# (2) The Divide-and-Regress Tree Structure:
# clf.tree:  {'split_var': 5, 'split_value': 6.939,
# 'L': {'split_var': 12, 'split_value': 14.37,
# 'L': Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None,
#     normalize=False, random_state=None, solver='auto', tol=0.001),
# 'R': Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None,
#     normalize=False, random_state=None, solver='auto', tol=0.001)},
# 'R': {'split_var': 5, 'split_value': 7.42,
# 'L': Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None,
#     normalize=False, random_state=None, solver='auto', tol=0.001),
# 'R': Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None,
#     normalize=False, random_state=None, solver='auto', tol=0.001)}}
#
# 2. This is what I got if spliting the train and test set of ratio 0.75:0.25, using the X_train and y_train to
# construct a decision tree, then use this decision tree to predict X_test. Here I use different Random State,
# and the MSE shows great difference(Below are two example, acutually I also tried several other random state value
# such as 40).
## Random State = 100 ##
# (1) Different Evaluation Methods:
# MSE: 39.699456985889526, RMSE: 6.300750509732117, RSQ: 0.6080419016969111,
# MAE: 3.4270435093569347, MAPE: 19.202325125739215, MedAE: 2.026471838791064
# (2) The Divide-and-Regress Tree Structure:
# clf.tree:  {'split_var': 5, 'split_value': 6.939,
# 'L': {'split_var': 12, 'split_value': 14.69,
# 'L': Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None,
#     normalize=False, random_state=None, solver='auto', tol=0.001),
# 'R': Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None,
#     normalize=False, random_state=None, solver='auto', tol=0.001)},
# 'R': {'split_var': 5, 'split_value': 7.42,
# 'L': Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None,
#     normalize=False, random_state=None, solver='auto', tol=0.001),
# 'R': Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None,
#     normalize=False, random_state=None, solver='auto', tol=0.001)}}
## Random State = 45 ##
# (1) Different Evaluation Methods:
# MSE: 14.399964861245298, RMSE: 3.7947285622617724, RSQ: 0.857886269059138,
# MAE: 2.621492944667934, MAPE: 13.939235909308648, MedAE: 2.0185557065580895
# (2) The Divide-and-Regress Tree Structure:
# clf.tree:  {'split_var': 12, 'split_value': 9.54,
# 'L': {'split_var': 5, 'split_value': 7.135,
# 'L': Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None,
#      normalize=False, random_state=None, solver='auto', tol=0.001),
# 'R': Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None,
#      normalize=False, random_state=None, solver='auto', tol=0.001)},
# 'R': {'split_var': 12, 'split_value': 19.15,
# 'L': Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None,
#      normalize=False, random_state=None, solver='auto', tol=0.001),
# 'R': Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None,
#      normalize=False, random_state=None, solver='auto', tol=0.001)}}

# 3. Interpret the selected splitting pairs:
# When decision tree is applied to regression tasks, we use MSE as the criteria for split pair
# selection. To minimize Mean Squared Error, we find the best splitting pair that minimizes:
# size_L * Var(y|Left) + size_R * Var(y|Right)
# My method for choosing all the possible values for a specific variable:
# Loop on all the possible values of a specific varaible and calculate the MSE of the specific
# variable and value, then choose the pair that produce the least MSE as split pair.
# And this is what I got for the first partition:
# 'split_var': 5, 'split_value': 6.939
# Becides, the great difference if we split the train and test set using different random state indicates that
# our model is a high "varaince" model.
