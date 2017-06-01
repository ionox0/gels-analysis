
# coding: utf-8

# In[1]:

# Import the modules
import cv2
from sklearn.externals import joblib
from sklearn import datasets
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn import preprocessing
import numpy as np
from collections import Counter


import keras
from keras import applications
from keras.models import Sequential
from keras.datasets import mnist
from keras.layers import Dense, Dropout, Flatten, Activation, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.utils.np_utils import to_categorical

from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.utils import to_categorical

from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV


from matplotlib import pyplot as plt
get_ipython().magic(u'matplotlib auto')


# In[2]:

img = cv2.imread('../data/gels_nov_2016/Im6 - p. 6.png', cv2.IMREAD_COLOR)

plt.imshow(img)


# In[ ]:



