# -*- coding: utf-8 -*-
"""
Created on SJuly 25 2017

@author: tcourtney
"""

# Classification template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values


# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X1 = LabelEncoder()
X[:, 1] = labelencoder_X1.fit_transform(X[:, 1])
labelencoder_X2 = LabelEncoder()
X[:, 2] = labelencoder_X2.fit_transform(X[:, 2])


onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

X=X[:,1:]


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


import keras
from keras.models import Sequential
from keras.layers import Dense
# Fitting classifier to the Training set
# Create your classifier here
classifier = Sequential()
classifier.add(Dense(output_dim=6, init = 'uniform', activation = 'relu', input_dim=11 ))

#adding the second hidden layer
classifier.add(Dense(output_dim=6, init = 'uniform', activation = 'relu'))

#adding the output layer
classifier.add(Dense(output_dim=1, init = 'uniform', activation = 'sigmoid'))


#Compiling the ANN
classifier.compile(optimizer = 'adam', loss='binary_crossentropy', metrics= ['accuracy'])
classifier.fit(X_train, y_train, batch_size=10, nb_epoch=100)



# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)