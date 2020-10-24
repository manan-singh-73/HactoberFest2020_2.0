#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import all required libraries
import os
import random
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.applications import VGG16
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Import the dataset
X, y = keras.datasets.cifar10.load_data()


# In[3]:


#Splitting the data into training and testing data
train_X = X[0]
train_y = X[1]
test_X = y[0]
test_y = y[1]


# In[4]:


# Printing the shapes of all to see if the dataset is correctly sorted
print(train_X.shape)
print(train_y.shape)
print(test_X.shape)
print(test_y.shape)


# ## Labels
#
# airplane : 0,
# automobile : 1,
# bird : 2,
# cat : 3,
# deer : 4,
# dog : 5,
# frog : 6,
# horse : 7,
# ship : 8,
# truck : 9

# In[5]:


# A random image from the training dataset
'''val = random.randint(0, 50000)
print("Index: ", val)
plt.title(train_y[val])
plt.imshow(train_X[val])


# In[6]:


# A random image from the training dataset
val = random.randint(0, 50000)
print("Index: ", val)
plt.title(train_y[val])
plt.imshow(train_X[val])


# In[7]:


# A random image from the testing dataset
val = random.randint(0, 10000)
print("Index: ", val)
plt.title(test_y[val])
plt.imshow(test_X[val])


# In[8]:


# A random image from the testing dataset
val = random.randint(0, 10000)
print("Index: ", val)
plt.title(test_y[val])
plt.imshow(test_X[val])'''


# In[9]:


# One-Hot Encoding the target labels
train_y_act = to_categorical(train_y, 10)
test_y_act = to_categorical(test_y, 10)


# In[11]:


num_of_classes = 10
epochs = 2
input_shape = (32, 32, 3)
batch_size = 32


# In[16]:


model = Sequential()


# In[18]:


# Adding layers to the Sequential Model to make a CNN
model.add(Conv2D(250, kernel_size = (1, 1), strides = (1, 1), activation = 'relu', input_shape = input_shape))
model.add(Conv2D(250, kernel_size = (1, 1), strides = (1, 1), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(500, kernel_size = (1, 1), strides = (1, 1), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(750, kernel_size = (1, 1), strides = (1, 1), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(1000, kernel_size = (1, 1), strides = (1, 1), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(1500, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(300, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(num_of_classes, activation = 'softmax'))


# In[20]:


#If model exists, uncomment this and execute
#model = tf.keras.models.load_model('Cifar_10_Model_TestAcc_54_46.h5')


# In[ ]:


#Else, uncomment these lines and run
model.compile(SGD(learning_rate = 0.01), loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_X, train_y_act, validation_split = 0.01, epochs = epochs, batch_size = batch_size, verbose=1, shuffle=1)


# In[ ]:


#history2 = mod_model.fit(train_X_act, train_y_act, validation_split = 0.01, epochs = epochs, batch_size = batch_size, verbose=1, shuffle=1)


# In[22]:


#Testing the model using the test data
result = mod_model.evaluate(test_X_act, test_y_act, batch_size = batch_size)
print("Test Loss: ", result[0])
print("Test Accuracy", result[1])


# In[23]:


#Saving the model
mod_model.save('Cifar_10_Model_TestAcc_56_11.h5')


# In[ ]:
