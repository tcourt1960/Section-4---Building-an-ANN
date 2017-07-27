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

#Adding the input layer and first hidden layer
classifier.add(Dense(output_dim=6, init = 'uniform', activation = 'relu', input_dim=11 ))

#adding the second hidden layer
classifier.add(Dense(output_dim=6, init = 'uniform', activation = 'relu'))

#adding the output layer
classifier.add(Dense(output_dim=1, init = 'uniform', activation = 'sigmoid'))


#Compiling the ANN
classifier.compile(optimizer = 'adam', loss='binary_crossentropy', metrics= ['accuracy'])

# Training the ANN to the Training Set
classifier.fit(X_train, y_train, batch_size=100, nb_epoch=100)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred=(y_pred>0.5)

# Making the Confusion Matrix on TEST set
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


'''make a prediction for a new customer
Geography = France
Credit Score : 600
Gender: Male
Age: 40
Tenure: 3
Balance: 60000
Number of products: 2
Has Credit Card: Yes
Is Active Member: Yes
estimated salary: 50000 '''

new_prediction = classifier.predict(sc.transform(np.array([[0,0, 600, 1, 40, 3, 60000, 2, 1,1, 5000]])))
print(new_prediction, (new_prediction>0.5))








