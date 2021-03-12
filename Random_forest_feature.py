# -*- coding: utf-8 -*-
"""
Created on Wed May 29 15:22:55 2019

@author: arnaldo
"""
import os
from sklearn.ensemble import RandomForestRegressor

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


feature_list = list(inputs.columns)


print(feature_list)

features =  np.array(inputs)
targets = np.array(activity)

# Split the data into training and testing sets
train_features, test_features, train_targets, test_targets = train_test_split(features, targets, test_size = 0.25, random_state = 42)


from sklearn.ensemble import RandomForestRegressor

# Import the model we are using

# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42, max_depth=10)

# Train the model on training data
rf.fit(train_features, train_targets);

# Use the forest's predict method on the test data
predictions = rf.predict(test_features)

# Calculate the absolute errors
errors = (abs(predictions - test_targets))

for i in range(len(predictions)):
    
    print (test_targets[i])
    
print ("stop")

for i in range(len(predictions)):
    
    print (predictions[i])

# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'units.')

x = np.arange(len(predictions))

plt.plot( x, test_targets,    marker='', color='blue', linewidth=2, linestyle='dashed', label="actual")
plt.plot( x, predictions, marker='', color='red', linewidth=2, label='predicted')

plt.legend()

plt.show()

# Pull out one tree from the forest
tree = rf.estimators_[5]

# Export the image to a dot file
export_graphviz(tree, out_file = 'single_tree.dot', feature_names = feature_list, rounded = True)

# Use dot file to create a graph
(graph, ) = pydot.graph_from_dot_file('single_tree.dot')

# Write graph to a png file
graph.write_png('single_tree.png')

# Get numerical feature importances
importances = list(rf.feature_importances_)

# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]

# Sort the feature importances by most important first
feature_importances = sorted(feature_importances)

for line in range(len(feature_importances)):
    print (feature_importances[line])

# list of x locations for plotting
x_values = list(range(len(importances)))

# Make a bar chart
plt.bar(x_values, importances, orientation = 'vertical', color = 'r', edgecolor = 'k', linewidth = 1.2)

# Tick labels for x axis
plt.xticks(x_values, feature_list, rotation='vertical')

# Axis labels and title
plt.ylabel('Importance'); plt.xlabel('Variable');
plt.title('Variable Importances');
plt.legend()

plt.show()
