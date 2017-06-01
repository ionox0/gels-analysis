
# coding: utf-8

# In[23]:

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


# ### https://arxiv.org/pdf/1702.00723.pdf

# In[24]:

# Load the dataset
dataset = datasets.fetch_mldata("MNIST Original")

# Extract the features and labels
features = np.array(dataset.data, 'int16')
labels = np.array(dataset.target, 'int')

# Extract the hog features
list_hog_fd = []
for feature in features:
    fd = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
    list_hog_fd.append(fd)
hog_features = np.array(list_hog_fd, 'float64')

# Normalize the features
pp = preprocessing.StandardScaler().fit(hog_features)
hog_features = pp.transform(hog_features)
print ("Count of digits in dataset", Counter(labels))


# ### SVC

# In[5]:

clf = SVC(kernel='rbf')
clf.fit(hog_features, labels)

# Save the classifier
joblib.dump((clf, pp), "digits_cls.pkl", compress=3)


# ### CNN

# In[27]:

def get_model():
    model = Sequential()

    model.add(Conv2D(32, (3,3), input_shape=(28,28,1), padding='same'))
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
    model.add(Dense(10))
    model.add(Activation('softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    return model


batch_size = 128
num_classes = 10
epochs = 7

model = get_model()

# Load the dataset
dataset = datasets.fetch_mldata("MNIST Original")

# Extract the features and labels
features = np.expand_dims(features.reshape(70000,28,28), axis=3)
labels = to_categorical(np.array(dataset.target, 'int'))

history_bn = model.fit(features, labels,
      batch_size=batch_size,
      epochs=epochs)

# Save the classifier
model.save(filepath='./digits_cnn')


# In[43]:

plt.imshow(np.array(dataset.data, 'int16'))


# ### Classify Digits

# In[4]:

from skimage.filters import threshold_otsu

def do_threshold(img):
    thresh = threshold_otsu(img)
#     thresh = threshold_adaptive(image, 15, 'mean')
    binary = img > thresh
    return np.array(binary)

im = cv2.imread('../data/gels_nov_2016/Im{} - p. {}.png'.format(41, 41))
# Threshold to get just the digits
# img_thresholded = do_threshold(im)
img_thresholded = im[:,:,0] < 80

print(img_thresholded.shape)

img_thresholded = img_thresholded[700:900,280:900]
reg = im[700:900,280:900]
im = img_thresholded.astype(np.uint8)

im.shape


# ### HOG

# In[21]:

clf, pp = joblib.load("digits_cls.pkl")
model = keras.models.load_model('./digits_cnn')

imgs = [1,6,7,12,21,22,41,42,51,52,56,83,84,89,90,96,97,106,123,131,136,152,153,156,157]

for i in [41]:
#     im = cv2.imread('../data/gels_nov_2016/Im{} - p. {}.png'.format(i, i))
    
#     # Convert to grayscale and apply Gaussian filtering
#     im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
#     im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)

#     # Threshold the image
#     ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)

    # Find contours in the image
    im_th = im.copy().astype(np.uint8)
    ctrs = cv2.findContours(im_th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get rectangles contains each contour
    rects = [cv2.boundingRect(ctr) for ctr in ctrs[1]]

    # For each rectangular region, calculate HOG features and predict
    # the digit using classifier.
    for rect in rects:
        print('here')
        # Draw the rectangles
        cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (255, 255, 255), 1)

        # Make the rectangular region around the digit
        leng = int(rect[3] * 1.6)
        pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
        pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
        roi = im_th[pt1:pt1+leng, pt2:pt2+leng]

        if roi.shape[0] == 0 or roi.shape[1] == 0: continue
        # Resize the image
        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        roi = cv2.dilate(roi, (3, 3))

        # Calculate the HOG features
        roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
        roi_hog_fd = pp.transform(np.array([roi_hog_fd], 'float64'))
        
        roi_cnn = np.expand_dims(roi, axis=2)
        # Choose model here
#         nbr = model.predict(np.array([roi_cnn]))
        nbr = clf.predict(roi_hog_fd)
#         nbr = [np.argmax(nbr)]

        cv2.putText(im, str(int(nbr[0])), (rect[0], rect[1]), cv2.FONT_ITALIC, 1, (255, 255, 255), 3)
        # (img, text, org, fontFace, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]]) → None

    plt.imshow(im)
#     plt.imshow(reg)
    plt.show()


# In[39]:

plt.imshow(reg)


# In[40]:

plt.imshow(im)


# In[37]:

np.expand_dims(features.reshape(70000,28,28), axis=3).shape


# ### MSER

# In[3]:

img = cv2.imread('../data/gels_nov_2016/Im2 - p. 2.png', cv2.IMREAD_GRAYSCALE)

# img = img[:,:,2] < 100
# img = img < 60
# img = img.astype(np.uint8)

from skimage import img_as_ubyte

img = img_as_ubyte(img)

mser = cv2.MSER_create()
mser_areas = mser.detect(img)


# In[4]:

for area in mser_areas:
    pt = area.pt
    x = int(pt[0])
    y = int(pt[1])
    size = int(area.size)
    cv2.rectangle(img, (x - size, y - size), (x + size, y + size), (255, 255, 255), 1)
    
plt.imshow(img)


# In[5]:

mser = cv2.MSER_create()
mser_areas = mser.detect(img)

regions, _ = mser.detectRegions(img)
hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]

cv2.polylines(img, hulls, 1, (255, 255, 255))

plt.imshow(img)


# In[ ]:

roi = img[hulls[0]]
print(roi.shape)
roi = cv2.resize(roi, (28,28))
# print(roi.shape)
# model.predict(roi)


# ### Connected Comps

# In[52]:

import scipy
from scipy import ndimage
import matplotlib.pyplot as plt

fname='../data/gels_nov_2016/Im{} - p. {}.png'.format(41, 41)
blur_radius = 0.2
threshold = 50

img = scipy.misc.imread(fname)[700:900,280:900] # gray-scale image
print(img.shape)

# smooth the image (to remove small objects)
imgf = ndimage.gaussian_filter(img, blur_radius)
threshold = 200

# find connected components
labeled, nr_objects = ndimage.label(imgf > threshold) 
print "Number of objects is %d " % nr_objects

plt.imshow(labeled.astype(float))

plt.show()


# ### CC #2

# In[22]:

from skimage import measure
try:
    from skimage import filters
except ImportError:
    from skimage import filter as filters
import matplotlib.pyplot as plt
import numpy as np
import scipy


fname='../data/gels_nov_2016/Im{} - p. {}.png'.format(41, 41)
n = 12
l = 256
np.random.seed(1)

im = scipy.misc.imread(fname)[700:900,280:900]
im = im.astype(np.float32)
im = im[:,:,0]
blobs = im > 0.7 * im.mean()
blobs = blobs.astype(np.float32)

all_labels = measure.label(blobs).astype(np.float32)
blobs_labels = measure.label(blobs, background=0).astype(np.float32)
plt.imshow(blobs_labels)


# In[16]:

im = im.astype(np.uint8)


# In[ ]:



