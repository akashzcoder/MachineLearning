#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 21:01:46 2017

@author: akashsingh
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('Data.csv')
#now decide the matrix of feature vector and dependent vactor 
X= data.iloc[:,:-1].values # feature vector
Y= data.iloc[:,3] #dependent variable vector
# preparing the data - processing the missing data
# 1. remove the lines that have missing data, but this can be dangerous
# 2. take the mean of the column
from sklearn.preprocessing import Imputer
# imputer class allows us to take care of the missing data
# create an object of the class Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
# fit this imputer object to our dataset
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:, 1:3])
# its 3 at the end because the upper bound is excluded
# after fixing the missing variable, we must encode the categorial variable
# library to encode the categorial data is also sklearn library
# but we will use the calss LabelEncoder
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:, 0])
# but putting it as 0, 1 and 2 means it might consider as 2>1>0, but they were categorial and not comparitive
# for this we will have to use OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)
# Splitting dataset into training set and test set
# performance on the test set should not be very different from that in the training set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2, random_state = 1)
# next step:
    #Feature Scaling : feature scaling is very important because most of the machine learning model is based upon the Euclidean distance
    # feature scaling is performed via: Standardization and normalization
# standardization: (x-mean(x))/(standard deviation)
# normalization = ((x-min)/(max-min))
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
# do we need to fit and transform dummy variables? it depends on the context, but it wont break your model
# even if the algorithms are not based on the euclidean distance the model will converge a lot faster if the features are scaled
# in classification we do not scale the dependent variable but in regression we will have to scale it.
