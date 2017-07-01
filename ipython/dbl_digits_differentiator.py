
# coding: utf-8

# In[6]:

import matplotlib
matplotlib.use('Agg')

import os
import cv2
import copy
import scipy
import pprint
import random
import itertools
import numpy as np
from random import randint
from sklearn.svm import SVC
from sklearn import datasets
from collections import Counter
from skimage.feature import hog
from skimage.morphology import square, disk
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import black_tophat, skeletonize, convex_hull_image
from skimage.morphology import disk

from sklearn import preprocessing
from skimage import data
import keras
from sklearn.cross_validation import train_test_split
from keras.utils.np_utils import to_categorical
from sklearn.externals import joblib


# In[2]:

from matplotlib import pyplot as plt

get_ipython().magic(u'matplotlib auto')


# In[3]:

def plot_things(things, labels):
    count = len(things)
    plt.figure(figsize=(20, 20))
    
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    for i, thing in enumerate(things):
        cols = 10
        rows = int(count / cols) + 1
        ax = plt.subplot(rows, cols, 1 + i)
        
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_title(labels[i])

        plt.imshow(thing)
    plt.show()


# In[10]:

from sklearn.datasets import fetch_mldata

mnist = fetch_mldata('MNIST original')
data = mnist.data
target = mnist.target


# # Singles

# In[13]:

single_labels = np.zeros(data.shape[0])
single_labels = to_categorical(single_labels)


# # DBLs

# In[ ]:

import random
from skimage.filters import threshold_otsu

def random_combination(iterable, r):
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(random.sample(xrange(n), r))
    return tuple(pool[i] for i in indices)

def remove_border(img):
    where = np.where(img > 0)
    y1 = np.min(where[0])
    y2 = np.max(where[0])
    x1 = np.min(where[1])
    x2 = np.max(where[1])
    crop = img[y1:y2 , x1:x2]
    return crop

n_samples = 1000000
combs = []
combs_labels = []
data_w_labels = zip(data, target)
random.shuffle(data_w_labels)

for i in range(n_samples):
    comb = random_combination(data_w_labels, 2)
    
    first_digit = comb[0][0].reshape((28, 28))
    second_digit = comb[1][0].reshape((28, 28))

#     thresh = threshold_otsu(first_digit)
#     first_digit = (first_digit > thresh).astype(np.uint8)
    
#     thresh = threshold_otsu(second_digit)
#     second_digit = (second_digit > thresh).astype(np.uint8)
    
    first_mod = remove_border(first_digit)
    second_mod = remove_border(second_digit)
    
    # Make sure height diff is divisible by 2
    if not (first_mod.shape[0] % 2 == 0):
        first_mod = np.vstack([first_mod, np.zeros((1, first_mod.shape[1]))])
    if not (second_mod.shape[0] % 2 == 0):
        second_mod = np.vstack([second_mod, np.zeros((1, second_mod.shape[1]))])
    
    height_diff = first_mod.shape[0] - second_mod.shape[0]
    
    if height_diff < 0:
        padding = int(-height_diff / 2.0)
        thepad = np.zeros((padding, first_mod.shape[1]))
        first_mod = np.vstack([first_mod, thepad])
        first_mod = np.vstack([thepad, first_mod])
    elif height_diff > 0:
        padding = int(height_diff / 2.0)
        thepad = np.zeros((padding, second_mod.shape[1]))
        second_mod = np.vstack([second_mod, thepad])
        second_mod = np.vstack([thepad, second_mod])
    
    # Align width
    overlap = 1
    
    height = first_mod.shape[0]
    width_1 = first_mod.shape[1]
    width_2 = second_mod.shape[1]
    
    padding = np.zeros((height, width_2 - overlap))
    first_mod = np.hstack([first_mod, padding])
    padding = np.zeros((height, width_1 - overlap))
    second_mod = np.hstack([padding, second_mod])
    
    overlapped = first_mod.astype(np.uint64) + second_mod.astype(np.uint64)
    overlapped = np.clip(overlapped, 0, 255).astype(np.uint8)
    padded = np.pad(overlapped, (5, 5), 'constant', constant_values=(0, 0))  
    
#     binary = (padded > 0).astype(np.uint8)
    combs.append(padded)
    
    label = int(  str(int(comb[0][1])) + str(int(comb[1][1]))  )
    combs_labels.append( 1 )
    
# combs = np.array(combs)
combs_labels = np.array(combs_labels)


# In[ ]:

combs_reshaped = [cv2.resize(x, (38, 28)) for x in combs]


# In[ ]:

i = 50
plot_things(combs_reshaped[i:i + 50], combs_labels[i:i + 50])


# # Train Classifier

# In[ ]:

# Extract the features and labels
combs_np = np.array(combs_reshaped)
dbl_features = np.expand_dims(combs_np.reshape(1000000, 28, 38), axis=3)
dbl_labels = to_categorical(combs_labels)

dbl_single = np.concat([data, dbl_features], axis=0)
dbl_single_labels = np.concat([single_labels, dbl_labels], axis=0)


# In[ ]:

import keras
from keras import applications
from keras.models import Sequential
from keras.datasets import mnist
from keras.layers import Dense, Dropout, Flatten, Activation, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K


def get_model():
    model = Sequential()

    model.add(Conv2D(32, (3,3), input_shape=(28, 38, 1), padding='same'))
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


# In[ ]:

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
model.save(filepath='./double_digits_cnn_nonbinary')

