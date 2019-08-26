#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#load dataset
dataset = pd.read_csv('mushrooms.csv')
dataset.head()
#dataset.describe()
#dataset.info()


# In[ ]:


#assign X and y
X = dataset.iloc[:, [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]].values
y = dataset.iloc[:, [0]].values
#X = pd.DataFrame(X)
#y = pd.DataFrame(y)


# In[ ]:


# Encoding categorical data for X dummy variables (not optimized)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.utils import np_utils
encoder = LabelEncoder()
X[:, 1] = encoder.fit_transform(X[:, 1])
X[:, 2] = encoder.fit_transform(X[:, 2])
X[:, 0] = encoder.fit_transform(X[:, 0])
X[:, 3] = encoder.fit_transform(X[:, 3])
X[:, 4] = encoder.fit_transform(X[:, 4])
X[:, 5] = encoder.fit_transform(X[:, 5])
X[:, 6] = encoder.fit_transform(X[:, 6])
X[:, 7] = encoder.fit_transform(X[:, 7])
X[:, 8] = encoder.fit_transform(X[:, 8])
X[:, 9] = encoder.fit_transform(X[:, 9])
X[:, 10] = encoder.fit_transform(X[:, 10])
X[:, 11] = encoder.fit_transform(X[:, 11])
X[:, 12] = encoder.fit_transform(X[:, 12])
X[:, 13] = encoder.fit_transform(X[:, 13])
X[:, 14] = encoder.fit_transform(X[:, 14])
X[:, 15] = encoder.fit_transform(X[:, 15])
X[:, 16] = encoder.fit_transform(X[:, 16])
X[:, 17] = encoder.fit_transform(X[:, 17])
X[:, 18] = encoder.fit_transform(X[:, 18])
X[:, 19] = encoder.fit_transform(X[:, 19])
X[:, 20] = encoder.fit_transform(X[:, 20])
X[:, 21] = encoder.fit_transform(X[:, 21])


# In[ ]:


onehotencoder = OneHotEncoder(categorical_features ='all')
X = onehotencoder.fit_transform(X).toarray()
#X = onehotencoder.fit_transform(X[1].reshape(1,-1)).toarray()
X = X[:, 1:]
X = pd.DataFrame(X)
#pd.get_dummies(X)
X.head()
#X[1]


# In[ ]:


#encoding for y
labelencoder_y_0 = LabelEncoder()
y[:, 0] = labelencoder_y_0.fit_transform(y[:, 0])

#split data -try the model with random state 42 and also it is checked with random state 0
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)
#X_train.shape

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[ ]:


yprime=pd.DataFrame(y)
yprime.head(10).T


# In[ ]:


import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import os
classifier = Sequential()
# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 12, init = 'uniform', activation = 'relu', input_dim = 116))
# Adding the second hidden layer
classifier.add(Dense(output_dim = 12, init = 'uniform', activation = 'relu'))
# Adding the third hidden layer
classifier.add(Dense(output_dim = 10, init = 'uniform', activation = 'relu'))
# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.fit(X_train, y_train, batch_size = 50, nb_epoch = 3)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = y_pred.round().astype(int)
y_test = y_test.astype(int)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
scores = classifier.evaluate(X_train, y_train) 
print("%s: %.2f%%" % (classifier.metrics_names[1], scores[1]*100))
print(cm)


# In[ ]:


#Save the model
classifier_json = classifier.to_json() 
with open("classifier.json", "w") as json_file:
    json_file.write(classifier_json) 
# serialize weights to HDF5 
classifier.save_weights("classifier.h5") 
print("Saved model to disk")


# In[ ]:


json_file = open( 'classifier.json' , 'r' )
loaded_classifier_json = json_file.read() 
json_file.close() 
loaded_classifier = model_from_json(loaded_classifier_json) 
# load weights into new model 
loaded_classifier.load_weights("classifier.h5") 
print("Loaded model from disk")


# In[ ]:


# new instances where we do not know the answer

# from sklearn.datasets.samples_generator import make_blobs
#from sklearn.preprocessing import MinMaxScaler
#scalar = MinMaxScaler().fit(X_train)
#Xnew, _ = make_blobs(n_samples=3, centers=2, n_features=24, random_state=1)
#Xnew = scalar.transform(Xnew)
req0 = np.array(['x','s','n','t','p','f','c','n','k','e','e','s','s','w','w','p','w','o','p','k','s','u'])
req1=np.array(['x', 's', 'y', 't', 'a', 'f', 'c', 'b', 'k', 'e', 'c', 's', 's',
       'w', 'w', 'p', 'w', 'o', 'p', 'n', 'n', 'g'])
req2=np.array(['b', 's', 'w', 't', 'l', 'f', 'c', 'b', 'n', 'e', 'c', 's', 's',
       'w', 'w', 'p', 'w', 'o', 'p', 'n', 'n', 'm'])
req5=np.array(['x', 'y', 'y', 't', 'a', 'f', 'c', 'b', 'n', 'e', 'c', 's', 's',
       'w', 'w', 'p', 'w', 'o', 'p', 'k', 'n', 'g'])

input_array=dataset.iloc[:, [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]].values
new_array=np.vstack([input_array, req0])
#new_array[8124]

#encoder = LabelEncoder()
#req = encoder.fit_transform(req1)
#test=req.reshape(1,-1)
encoder = LabelEncoder()
new_array[:, 1] = encoder.fit_transform(new_array[:, 1])
new_array[:, 2] = encoder.fit_transform(new_array[:, 2])
new_array[:, 0] = encoder.fit_transform(new_array[:, 0])
new_array[:, 3] = encoder.fit_transform(new_array[:, 3])
new_array[:, 4] = encoder.fit_transform(new_array[:, 4])
new_array[:, 5] = encoder.fit_transform(new_array[:, 5])
new_array[:, 6] = encoder.fit_transform(new_array[:, 6])
new_array[:, 7] = encoder.fit_transform(new_array[:, 7])
new_array[:, 8] = encoder.fit_transform(new_array[:, 8])
new_array[:, 9] = encoder.fit_transform(new_array[:, 9])
new_array[:, 10] = encoder.fit_transform(new_array[:, 10])
new_array[:, 11] = encoder.fit_transform(new_array[:, 11])
new_array[:, 12] = encoder.fit_transform(new_array[:, 12])
new_array[:, 13] = encoder.fit_transform(new_array[:, 13])
new_array[:, 14] = encoder.fit_transform(new_array[:, 14])
new_array[:, 15] = encoder.fit_transform(new_array[:, 15])
new_array[:, 16] = encoder.fit_transform(new_array[:, 16])
new_array[:, 17] = encoder.fit_transform(new_array[:, 17])
new_array[:, 18] = encoder.fit_transform(new_array[:, 18])
new_array[:, 19] = encoder.fit_transform(new_array[:, 19])
new_array[:, 20] = encoder.fit_transform(new_array[:, 20])
new_array[:, 21] = encoder.fit_transform(new_array[:, 21])

onehotencoder = OneHotEncoder(categorical_features = 'all')
test = onehotencoder.fit_transform(new_array).toarray()
test = test[:, 1:]
test.shape


# In[ ]:


#make a prediction
ynew = classifier.predict_classes(test)
# show the inputs and predicted outputs
for i in range(len(test)):
    print("X=%s, Predicted=%s" % (test[i], ynew[i]))


# In[ ]:




