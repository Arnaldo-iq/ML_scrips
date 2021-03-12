#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 15:43:23 2019

@author: arnaldo
"""

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

place = "/home/pczaf/scripts/data_new.csv"

df = pd.read_csv(place, names=['PIC50','CHROM_LOGD_74','CHROM_LOGD_105','CHROM_LOGD_20','AMEMPERM_705','CLND_SOL', 'CLND_CONC', 'target'])


from sklearn.preprocessing import StandardScaler

features = ['PIC50','CHROM_LOGD_74','CHROM_LOGD_105','CHROM_LOGD_20','AMEMPERM_705','CLND_SOL', 'CLND_CONC']


# Separating out the features

x = df.loc[:, features].values

# Separating out the target

y = df.loc[:,['target']].values

# Standardizing the features

x = StandardScaler().fit_transform(x)


from sklearn.decomposition import PCA

pca = PCA(n_components=2)

principalComponents = pca.fit_transform(x)


principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
             

finalDf = pd.concat([principalDf, df[['target']]], axis = 1)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)

targets = ['GROUP1', 'GROUP2', 'GROUP3','GROUP4', 'GROUP5']
colors = ['r', 'g', 'b', 'k', 'm']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['target'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()
print(pca.explained_variance_ratio_)
