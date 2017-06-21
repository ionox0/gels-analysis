
# coding: utf-8

# In[2]:

import cv2
import scipy
import pprint
import random
import itertools
import numpy as np
from sklearn.svm import SVC
from sklearn import datasets
from collections import Counter
from skimage.feature import hog
from sklearn import preprocessing
from skimage import data
import keras
from sklearn.externals import joblib

from matplotlib import pyplot as plt
get_ipython().magic(u'matplotlib auto')


# In[3]:

clf, pp = joblib.load("digits_cls.pkl")
model = keras.models.load_model('./digits_cnn')


# In[4]:

from keras.datasets import mnist

data = mnist.load_data()


# ### Generating 2-digit combinations

# In[5]:

def remove_border(img):
    ctrs = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x,y,w,h = cv2.boundingRect(ctrs[0])
    crop = img[y:y+h,x:x+w]
    resize = cv2.resize(crop, (28, 28))
    return resize


i = 0
combs = []
combs_labels = []
data_w_labels = zip(data[0][0], data[0][1])
for comb in itertools.combinations(data_w_labels, 2):
    if i >= 1000000: break
        
    first_digit = comb[0][0]
    second_digit = comb[1][0]
    
    first_mod = remove_border(first_digit)
    second_mod = remove_border(second_digit)
    
    overlap = 8
    
    padding = np.zeros((28, 28 - int(overlap)))
    first_mod = np.concatenate((first_mod, padding), axis=1)
    second_mod = np.concatenate((padding, second_mod), axis=1)
    
    overlapped = first_mod + second_mod
        
    combs.append(overlapped)
    combs_labels.append( int(str(comb[0][1]) + str(comb[1][1])) )
    i += 1
    
combs = np.array(combs)
combs_labels = np.array(combs_labels)


# In[6]:

data[0][0].shape, combs_labels.shape


# In[8]:

i = 999
plt.imshow(combs[i])
combs_labels[i]


# ### Try some classifying

# In[10]:

from sklearn.cross_validation import train_test_split
from keras.utils.np_utils import to_categorical

# Extract the features and labels
features = np.expand_dims(combs.reshape(1000000, 28, 48), axis=3)
labels = to_categorical(combs_labels)

x_train, y_train, x_test, y_test = train_test_split(features, labels)


# In[15]:

import keras
from keras import applications
from keras.models import Sequential
from keras.datasets import mnist
from keras.layers import Dense, Dropout, Flatten, Activation, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K


def get_model():
    model = Sequential()

    model.add(Conv2D(32, (3,3), input_shape=(28,48,1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(100))
    model.add(Activation('softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    return model


batch_size = 512
num_classes = 10
epochs = 7

model = get_model()


# In[17]:

earlyStopping=keras.callbacks.EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto')

history_bn = model.fit(
    features,
    labels,
    validation_split=0.05,
    batch_size=batch_size,
    callbacks=[earlyStopping], 
    epochs=epochs
)

# Save the classifier
model.save(filepath='./double_digits_cnn')


# 
