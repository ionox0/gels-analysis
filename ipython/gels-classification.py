
# coding: utf-8

# In[1]:

import os
import itertools
import sys
import cv2
import numpy as np
import pandas as pd
import scipy
import skimage
from skimage import data, filters
import matplotlib
from matplotlib import pyplot as plt


# In[2]:

xls = pd.read_excel('../data/2012-2017_labels.xlsx', sheetname='2016')


# In[3]:

xls.shape


# In[4]:

serum_cols = xls.columns.str.contains('SPEP')


# In[12]:

cols = xls.columns[serum_cols]
spreadsheet_top = xls.ix[:30, cols]
spreadsheet_top.shape


# ### Attempt to OCR gel dates

# In[1]:

import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.pipeline import Pipeline
from sklearn import datasets

from skimage import data, filters

from keras.datasets import mnist
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.utils.np_utils import to_categorical

from matplotlib import pyplot as plt


# In[1]:

def extract_images_from_pdf(filename):
    # https://nedbatchelder.com/blog/200712/extracting_jpgs_from_pdfs.html
    pdf = open(filename, "rb").read()

    startmark = "\xff\xd8"
    startfix = 0
    endmark = "\xff\xd9"
    endfix = 2
    i = 0

    njpg = 0
    while True:
        istream = pdf.find("stream", i)
        if istream < 0:
            break
        istart = pdf.find(startmark, istream, istream + 20)
        if istart < 0:
            i = istream + 20
            continue
        iend = pdf.find("endstream", istart)
        if iend < 0:
            raise Exception("Didn't find end of stream!")
        iend = pdf.find(endmark, iend - 20)
        if iend < 0:
            raise Exception("Didn't find end of JPG!")

        istart += startfix
        iend += endfix
        print("JPG %d from %d to %d" % (njpg, istart, iend))

        jpg = pdf[istart:iend]
        jpgfile = open("./april_2016_gels/jpg%d.jpg" % njpg, "wb")
        jpgfile.write(jpg)
        jpgfile.close()

        njpg += 1
        i = iend
        
extract_images_from_pdf('../data/GelsApr2016.pdf')


# In[11]:

img = data.imread('./april_2016_gels/jpg3.jpg', 0)
plt.imshow(img)


# In[12]:

plt.show()


# In[28]:

from keras.datasets import mnist


(x_train, y_train), (x_test, y_test) = mnist.load_data()

model = Sequential()
model.add(Dense(32, input_shape=img.shape))
model.add(Dense(64))

model.compile()
model.train


# In[ ]:




# In[ ]:



