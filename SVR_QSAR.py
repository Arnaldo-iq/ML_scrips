#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 15:34:39 2020

@author: pczaf
"""
import numpy as np
import pandas as pd
#
# Import rdkit stuff
#
import rdkit
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import PandasTools
#
# Import the machine learning tools 
#
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, cross_val_predict, cross_validate
from sklearn import metrics
from sklearn.metrics import explained_variance_score
#
#Import matplot for our figures
#
import matplotlib.pyplot as plt
#
# Load the dataframe from a .csv file, stores the smiles strings for our compound and their PIC50a ctivities 

def qsar(smi_mol):

    training_data = pd.read_csv('Train.csv')
    training_smiles = np.array(training_data['Parent_SMILES'])
    training_activity = np.array(training_data['Value'])
#
    def generate_FP_matrix(smiles):
        morgan_matrix = np.zeros((1,2048))
        l=len(smiles)
#    
        for i in range(l):
            try:
                compound = Chem.MolFromSmiles(smiles[i])
                fp = Chem.AllChem.GetHashedAtomPairFingerprintAsBitVect(compound, nBits = 2048) 
                fp = fp.ToBitString()
                matrix_row = np.array ([int(x) for x in list(fp)])
                morgan_matrix = np.row_stack((morgan_matrix, matrix_row))
#            
                if i%500==0:
                    percentage = np.round(0.1*(i/1),1)
                    print ('Calculating fingerprint', percentage,  '% done')
#       
            except:
                print ('problem with index', i)
        morgan_matrix = np.delete(morgan_matrix, 0, axis = 0)
#    
        print('\n')
        print('Morgan Matrix Dimension is', morgan_matrix.shape)
#    
        return morgan_matrix

    def generate_FP(smiles):
        matrix = np.zeros((1,2048))
        mol = Chem.MolFromSmiles(smiles)
        fp = Chem.AllChem.GetHashedAtomPairFingerprintAsBitVect(mol, nBits = 2048)
        fp = fp.ToBitString()
        matrix_row = np.array ([int(x) for x in list(fp)])
        matrix = np.row_stack((matrix, matrix_row))
        matrix = np.delete(matrix, 0, axis = 0)
        return matrix

    matrix_feature_training = generate_FP_matrix(training_smiles)
    feature_training =  np.array(matrix_feature_training)
    target_training = np.array(training_activity)

    matrix_feature_test = generate_FP(smi_mol)
    feature_test = np.array(matrix_feature_test)

# Define the regressor to be used in our model. Here a Supporting Vectot Machine
#is chosen. Hyperparameters were optimized ina different step througha grid search.
#
    regressor=SVR(kernel='rbf', C = 3, gamma='scale', epsilon = 0.01, max_iter=-1)
#
# Spit thre data into training and test set.
#

    regressor.fit(feature_training, target_training)
    prediction=regressor.predict(feature_test)
    return prediction
#

test_smiles = 'o1nc(c(c1CC)c1cc(c(cc1)c1[nH]ccn1)[C@@H]1CN[C@@H](CN1)Cc1onc(c1)C)C'

result = float(qsar(test_smiles))
print(result)
