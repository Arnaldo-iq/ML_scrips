# -*- coding: utf-8 -*-
"""
Created on Wed May 29 15:22:55 2019

@author: arnaldo
"""
import os

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import RandomizedSearchCV

# Using Skicit-learn to split data into training and testing sets

from sklearn.model_selection import train_test_split

# Import tools needed for visualization

from sklearn.tree import export_graphviz

import pydot
import numpy as np
import matplotlib.pyplot as plt

# Pandas is used for data manipulation

import pandas as pd

# read as a xls (excel apparently, which is nice)

molecular_data = pd.read_csv('gsk1_new.csv')
smiles = np.array(molecular_data['Parent_SMILES'])
activity = np.array(molecular_data['PIC50'])

inputs =  pd.read_csv('dumb_input.csv')
features =  np.array(inputs)
targets = np.array(activity)


train_features, test_features, train_targets, test_targets = train_test_split(features, targets, test_size = 0.25, random_state = 42)


# Number of trees in random forest
n_estimators = [int(num) for num in np.linspace(start = 200, stop = 1000, num = 2)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(num) for num in np.linspace(10, 100, num = 2)]
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

print(random_grid)

# First create the base model to tune
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 10, verbose=2, random_state=42)

# Fit the random search model
rf_random.fit(train_features, train_targets)

best_grid = (rf_random.best_params_)

print(best_grid)
