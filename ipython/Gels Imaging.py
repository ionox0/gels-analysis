
# coding: utf-8

# In[3]:

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

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cross_validation import train_test_split

import sys 
sys.path.insert(0, '/Users/ianjohnson/.virtualenvs/cv/lib/python2.7/site-packages/')

get_ipython().magic(u'matplotlib inline')


# ### Extract images from pdfs

# In[8]:

# https://nedbatchelder.com/blog/200712/extracting_jpgs_from_pdfs.html
pdf = file('../data/blue gels older format 2010.pdf', "rb").read()

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
    istart = pdf.find(startmark, istream, istream+20)
    if istart < 0:
        i = istream+20
        continue
    iend = pdf.find("endstream", istart)
    if iend < 0:
        raise Exception("Didn't find end of stream!")
    iend = pdf.find(endmark, iend-20)
    if iend < 0:
        raise Exception("Didn't find end of JPG!")
     
    istart += startfix
    iend += endfix
    print "JPG %d from %d to %d" % (njpg, istart, iend)
    jpg = pdf[istart:iend]
    jpgfile = file("jpg%d.jpg" % njpg, "wb")
    jpgfile.write(jpg)
    jpgfile.close()
     
    njpg += 1
    i = iend


# In[9]:

def jpg(filename):
    return 'jpg' in filename

raw_imgs = []
for img_file in filter(jpg, os.listdir('.')):  
    img = data.imread(img_file, 0)
    img = cv2.rotate(img, rotateCode=2)
    raw_imgs.append(img)


# ### Vis an image

# In[10]:

plt.figure(figsize=(10, 10))
plt.imshow(raw_imgs[0])
plt.show()


# ### Extracting and visualizing the individual lanes

# In[11]:

def extract_lanes(gel_image, y_offset, x_offset, width, w_space, height, h_space):
    imgs = []
    for i in range(0, 20):
        x = x_offset + (i * width) + (i * w_space)
        y = y_offset
        imgs.append(img[y : y + height, x : x + width])
    return imgs

def plot_lanes(imgs):
    count = len(imgs)
    plt.figure(figsize=(20, 20))
    for i, img in enumerate(imgs):
        cols = 20
        rows = int(count / cols) + 1
        plt.subplot(rows, cols, 1 + i)
        plt.imshow(img)
        plt.title(i)
        plt.axis('off')


# In[12]:

y_offset = 300 #, 800
y_offset_2 = 800
x_offset = 100
x_offset_2 = 100
width = 80
w_space = 5
height = 280
h_space = 50

all_lanes = []
for img in raw_imgs:
    lanes = extract_lanes(
        img,
        y_offset,
        x_offset,
        width,
        w_space,
        height,
        h_space
    )
    all_lanes += lanes
    
    lanes = extract_lanes(
        img,
        y_offset_2,
        x_offset_2,
        width,
        w_space,
        height,
        h_space
    )
    all_lanes += lanes


# In[13]:

len(all_lanes)


# In[26]:

plt.imshow(raw_imgs[0][300:700, 100:1800])


# In[89]:

from skimage.filters import threshold_otsu
    
roi_metadata = {
    'x_start': 100,
    'x_end': 1800,
    'y_start': 300,
    'y_end': 600,
}


def extract_roi(image, roi_metadata):
    roi = image[
        roi_metadata['y_start'] : roi_metadata['y_end'],
        roi_metadata['x_start'] : roi_metadata['x_end']
    ]
    plt.imshow(roi)
    return roi

def calc_img_vertical_sum(img):
    vert_sum = img.sum(axis=0)
    return vert_sum

def do_threshold(img):
    thresh = threshold_otsu(img)
    binary = img > thresh
    return binary

def isolate_lanes(img):
    plt.imshow(img)
    vert_sum_img = calc_img_vertical_sum(img)    
    img_rolled = np.rollaxis(vert_sum_img, -1)
    separator_indices = np.where(img_rolled[2] == 300)[0]
    
    lanes = []
    for i, val in enumerate(separator_indices):
        if i == 0: continue
        if separator_indices[i] != separator_indices[i - 1] + 1:
            lanes.append(img[: , separator_indices[i - 1] : val])
    return lanes


img_roi = extract_roi(raw_imgs[0], roi_metadata)
roi_thresholded = do_threshold(img_roi)
lanes = isolate_lanes(roi_thresholded)


# In[90]:

len(lanes)


# In[100]:

plt.imshow(lanes[19])


# ### Best thresholding method

# In[50]:

import matplotlib
import matplotlib.pyplot as plt

from skimage import data
from skimage.filters import try_all_threshold

img = data.page()

fig, ax = try_all_threshold(img_roi, figsize=(10, 8), verbose=False)
plt.show()


# In[39]:

x = np.array([[1,2],[3,4]])
x.sum(axis=1)


# In[28]:

plot_lanes(all_lanes[40:160])


# ### Visualize the mean over all lanes

# In[43]:

plt.matshow(np.mean(lanes, axis=0)[:, :, 2], cmap="viridis")


# ### Mean across horizontal axis, along the vertical axis

# In[109]:

gels_means = []
for lane in all_lanes:
    lane_mean = lane.mean(axis=1)
    # Just take the blue channels
    gels_means.append(lane_mean[:,2])
    
gels_means_df = pd.DataFrame(gels_means)


# In[111]:

plt.figure(figsize=(20, 20))
gels_means_df.T.plot(alpha=.1, c='k', ax=plt.gca(), legend=None)


# ### Mean densitometry across all lanes

# In[113]:

np.mean(gels_means_df, axis=0).plot()


# ### PCA

# In[114]:

from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
X = scale(gels_means_df)
pca = PCA()
X_pca = pca.fit_transform(X)


# In[115]:

plt.semilogy(pca.explained_variance_ratio_)


# In[25]:

plt.plot(pca.components_[0])
plt.plot(pca.components_[1])
plt.plot(pca.components_[2])
plt.plot(pca.components_[3])


# In[26]:

plt.scatter(X_pca[:, 0], X_pca[:, 1])


# ### Detect Danger Lanes

# In[116]:

indices = [np.any(x[0:100] < 180) for x in gels_means]
maybe_dz = filter(lambda x: np.any(x[0:100] < 180), gels_means)
maybe_dz = pd.DataFrame(maybe_dz)
len(maybe_dz)


# In[117]:

plt.figure(figsize=(20, 20))
maybe_dz.T.plot(alpha=.1, c='k', ax=plt.gca(), legend=None)


# In[64]:

dz_lanes = itertools.compress(all_lanes, indices)
plot_lanes(list(dz_lanes))


# ### Classification

# In[118]:

dz_idx = np.where(indices)[0].tolist()


# In[132]:

fake_labels = [i % 2 for i in range(len(means))]
attempt_labels = [0]*len(means)

# My own (probably bad) labels
w_disease = [1,41,43,46,66,75,81,83,100,117,121,161,167,184,201,212,206,207,
             238,241,245,245,256,263,274,281,282,286,287,307,319,320]
             
for dz in dz_idx:
    attempt_labels[dz] = 1

x_train, x_test, y_train, y_test = train_test_split(gels_means_df, attempt_labels)

print(len(dz_idx) / float(len(attempt_labels)))

clf = RandomForestClassifier()
clf.fit(x_train, y_train)
clf.score(x_test, y_test)


# In[ ]:

import sys
sys.path.insert(0, '/Users/ianjohnson/.virtualenvs/gels-analysis/lib/python2.7/site-packages')
import six
print(six.__version__)
print(six.__file__)

# from keras.datasets import mnist
# from keras.layers.core import Dense, Dropout, Activation
# from keras.optimizers import SGD, Adam, RMSprop
# from keras.utils import np_utils
import tensorflow as tf


# ### Watershed segmentation

# In[28]:

from skimage import morphology
from skimage.morphology import watershed
from skimage.feature import peak_local_max

# Generate an initial image with two overlapping circles
x, y = np.indices((80, 80))
x1, y1, x2, y2 = 28, 28, 44, 52
r1, r2 = 16, 20
mask_circle1 = (x - x1) ** 2 + (y - y1) ** 2 < r1 ** 2
mask_circle2 = (x - x2) ** 2 + (y - y2) ** 2 < r2 ** 2
image = np.logical_or(mask_circle1, mask_circle2)
# Now we want to separate the two objects in image
# Generate the markers as local maxima of the distance
# to the background
from scipy import ndimage
distance = ndimage.distance_transform_edt(image)
local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)), labels=image)
markers = morphology.label(local_maxi)
labels_ws = watershed(-distance, markers, mask=image)


# In[ ]:



