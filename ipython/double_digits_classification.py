
# coding: utf-8

# In[1]:

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


# # Generating 2-digit combinations

# In[3]:

from sklearn.datasets import fetch_mldata

mnist = fetch_mldata('MNIST original')
data = mnist.data
target = mnist.target


# In[19]:

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
    combs_labels.append( label )
    
# combs = np.array(combs)
combs_labels = np.array(combs_labels)


# In[20]:

combs_reshaped = [cv2.resize(x, (38, 28)) for x in combs]


# In[21]:

len(combs_reshaped)


# In[18]:

i = 50
plot_things(combs_reshaped[i:i + 50], combs_labels[i:i + 50])


# # Train Classifier

# In[ ]:

from sklearn.cross_validation import train_test_split
from keras.utils.np_utils import to_categorical

# Extract the features and labels
combs_np = np.array(combs_reshaped)
features = np.expand_dims(combs_np.reshape(1000000, 28, 38), axis=3)
labels = to_categorical(combs_labels)

x_train, y_train, x_test, y_test = train_test_split(features, labels)


# In[23]:

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


# # Test on Gels

# In[4]:

clf, pp = joblib.load('digits_cls.pkl')
model = keras.models.load_model('./double_digits_cnn_nonbinary')
model_single = keras.models.load_model('./digits_cnn_new')


# In[5]:

imgs = [1,6,12,21,41,42,51,52,56,83,84,89,90,96,97,106,123,131,136,152,153,156,157] #7, 22
filenames = ['../data/gels_nov_2016/Im{} - p. {}.png'.format(i, i) for i in imgs]
images = [scipy.misc.imread(f) for f in filenames]

april_imgs = [f for f in os.listdir('../data/april_2016_gels_renamed') if not 'tore' in f]
april_filenames = ['../data/april_2016_gels_renamed/{}'.format(f) for f in april_imgs]
april_images = [scipy.misc.imread(f) for f in april_filenames]


# In[155]:

def thresh_img(im, thresh, blue_thresh=False):
    if blue_thresh:
        # RGB --> BGR (openCV style)
        im_bgr = im.copy()[:,:,::-1]
        hsv = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2HSV)
        # Define range of blue color in HSV
        lower_blue = np.array([110,50,50])
        upper_blue = np.array([130,255,255])
        # Threshold the HSV image to get only blue colors
        im_th = cv2.inRange(hsv, lower_blue, upper_blue)
        im_th = cv2.bitwise_and(im_c, im_c, mask=im_th)
        im_th = cv2.cvtColor(im_th, cv2.COLOR_BGR2GRAY)
    else:
        ret, im_th = cv2.threshold(im, thresh, 255, cv2.THRESH_BINARY)
        im_th = im_th.astype(np.uint8)
    return im_th


def preprocessing(im_gray, blur, brightness_inc, contrast_inc, dilation_size, erosion_size):
    # Gaussian blur
    if blur:
        im_gray = cv2.medianBlur(im_gray, blur, blur)
        
    # Brightness
    if brightness_inc:
        im_expanded = im_gray.astype(np.uint64) + brightness_inc
        im_gray = np.clip(im_expanded, 0, 255).astype(np.uint8)
    # Contrast
    if contrast_inc:
        im_expanded = im_gray.astype(np.uint64) * contrast_inc
        im_gray = np.clip(im_expanded, 0, 255).astype(np.uint8)
    
    # Dilation
    if dilation_size:
        selem = disk(dilation_size)
        im_gray = dilation(im_gray, selem)
    # Erosion
    if erosion_size:
        selem = disk(erosion_size)
        im_gray = erosion(im_gray, selem)
    return im_gray

        
def find_rects_ctrs(im_th):
    # Find Contours
    _, ctrs, hierarchy = cv2.findContours(im_th, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    # Filter based on hierarchy
    ctrs = [c for i, c in enumerate(ctrs) if hierarchy[0][i][-1] == -1]
    hierarchy = [h for i, h in enumerate(hierarchy[0]) if hierarchy[0][i][-1] == -1]
    # Get Contour bounding boxes
    rects = [cv2.boundingRect(ctr) for ctr in ctrs]
    # Remove small rects
    ctrs = [c for i, c in enumerate(ctrs) if rects[i][3] > 10]
    rects = [r for r in rects if r[3] > 10]
    return rects, ctrs


def separate_connected(rects, ctrs, im_gray):
    rects_result = []
    ctrs_result = []
    for i, rect in enumerate(rects):
        x_start = rect[0]
        y_start = rect[1]
        width = rect[2]
        height = rect[3]

        im_roi = im_gray[y_start : y_start + height, x_start : x_start + width]

        # Draw filled contours (removes overlapping items)
        mask = np.zeros((height, width)).astype(np.uint8)
        mask = cv2.drawContours(mask, ctrs, i, (255, 255, 255), cv2.FILLED, offset=(-x_start, -y_start))
        roi = cv2.bitwise_and(mask.astype(np.uint8), im_roi.astype(np.uint8)).astype(np.uint8)

        cnt = ctrs[i]
        hull = cv2.convexHull(cnt, returnPoints = False)
        defects = cv2.convexityDefects(cnt, hull)

        mean_defect_dist = sum([d for s,e,f,d in defects[:,0]]) / defects.shape[0]
        large_defects = [(s,e,f,d) for s,e,f,d in defects[:,0] if d > mean_defect_dist]

        roi = np.pad(roi, ((y_start, 0), (x_start, 0)), 'constant', constant_values=(0,0))
        for s, e, f, d in large_defects:
            dists = [(cnt[f][0][0] - cnt[x][0][0])**2 + (cnt[f][0][1] - cnt[x][0][1])**2  if x != f else 999 for s, e, x, d in defects[:,0]]
            closest_idx = np.argmin(dists)
            closest = defects[closest_idx, :]
            start = tuple(cnt[closest[0][2]][0])
            end = tuple(cnt[f][0])
            cv2.line(roi, start, end, [0, 0, 0], 2)

        # Resegment
        new_rects, new_ctrs = find_rects_ctrs(roi)
        rects_result += new_rects
        ctrs_result += new_ctrs
        
    return rects_result, ctrs_result


def sort_rects(rects_ctrs):
    rects_sort = sorted(rects_ctrs, key=lambda r: r[0][0])
    
    result = []
    for rect in rects_sort:
        if rect[0][3] < 10:
            continue

        # Filter to only rects with overlap
        prev = rect
        first = rect
        rects_sort_filt = []
        rects_sort_filt.append(prev)
        for x in rects_sort:
            if x == first:
                continue
            if calc_overlap(x[0], prev[0]) > (prev[0][3] / 3.0) and horiz_dist_ratio_check(prev[0], x[0]):
                rects_sort_filt.append(x)
                prev = x
                
        result += rects_sort_filt
        for rect_sorted in rects_sort_filt:
            rects_sort.remove(rect_sorted)
            
    return zip(*result)


def calc_overlap(rect_1, rect_2):
    rect_1_x = rect_1[0]
    rect_1_y = rect_1[1]
    rect_1_width = rect_1[2]
    rect_1_height = rect_1[3]
    
    rect_2_x = rect_2[0]
    rect_2_y = rect_2[1]
    rect_2_width = rect_2[2]
    rect_2_height = rect_2[3]
    
    overlap = min(rect_1_y + rect_1_height, rect_2_y + rect_2_height) - max(rect_1_y, rect_2_y)
    return overlap


def horiz_dist_ratio_check(r1, r2):
    return abs(r1[0] + r1[2] - r2[0]) < (3 * r1[2])


SZ = 28
affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR
def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11'] / m['mu02']
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (SZ, SZ), flags=affine_flags)
    return img


# In[160]:

def extract_numbers(
    im,
    thresh=30,
    blue_thresh=False,
    separate_c=False,
    blur=0,
    contrast_inc=0,
    brightness_inc=0,
    opening_shape=None,
    closing_shape=None,
    dilation_size=0,
    erosion_size=0,
    should_deskew=False):
    
    im_c = im.copy()
    # Convert to grayscale
    im_gray = cv2.cvtColor(im_c, cv2.COLOR_BGR2GRAY)
    # Just blue channel
    im_one = im.copy().astype(np.uint8)[:,:,2]
    # Invert the image (todo - use im_gray?)
    im_gray = (255 - im_one)
    
    if blue_thresh:
        im_th = thresh_img(im, thresh, blue_thresh=True)
    else:
        im_th = thresh_img(im_gray, thresh, blue_thresh=False)
        
    # Opening / Closing (also for contours)
    if hasattr(opening_shape, 'shape'):
        im_th = opening(im_th, opening_shape)
    if hasattr(closing_shape, 'shape'):
        im_th = closing(im_th, closing_shape)
        
    im_gray = preprocessing(im_gray, blur, brightness_inc, contrast_inc, dilation_size, erosion_size)
        
    rects, ctrs = find_rects_ctrs(im_th)
    
    # Separate connected digits
    if separate_c:
        new_rects, new_ctrs = separate_connected(rects, ctrs, im_gray)
        rects += new_rects
        ctrs += new_ctrs
        
    # Sorted order bounding boxes
    sorted_rects, sorted_ctrs = sort_rects(zip(rects, ctrs))
    
    rois = []
    probs = []
    date_possibs = []
    cur_date_possibs = []
    prev_not_one_width = 99999
    prev_end_x = sorted_rects[0][0] + sorted_rects[0][2]
    prev_end_y = sorted_rects[0][1] + sorted_rects[0][3]
    
    # For each rectangular region, predict the digit using classifier
    for i, rect in enumerate(sorted_rects):
        x_start = rect[0]
        y_start = rect[1]
        width = rect[2]
        height = rect[3]

        # Skip short artifacts
        if height < 10: continue
        # Skip long artifacts
#         if width > 100: continue

        # Try with either im_gray, or im_th_all_colors here
        im_roi = im_gray[y_start : y_start + height, x_start : x_start + width]
        
        # Draw filled contours (removes overlapping items)
        mask = np.zeros((height, width)).astype(np.uint8)
        mask = cv2.drawContours(mask, sorted_ctrs, i, (255, 255, 255), cv2.FILLED, offset=(-x_start, -y_start))
        roi = cv2.bitwise_and(mask.astype(np.uint8), im_roi.astype(np.uint8)).astype(np.uint8)

        # Pad ROI to square
        if height < width:
            padding = int((width - height) / 2.0)
            roi = np.pad(roi, ((padding, padding), (0,0)), 'constant', constant_values=(0, 0))
        elif width < height:
            padding = int((height - width) / 2.0)
            roi = np.pad(roi, ((0,0), (padding, padding)), 'constant', constant_values=(0, 0))
        roi_pad = np.pad(roi, (5, 5), 'constant', constant_values=(0, 0))
        
        # Create new date possib (for newlines)
        if abs(x_start - prev_end_x) < 80 and abs(y_start - prev_end_y) < 80:
            newline = False
        else:
            newline = True
        if newline:
            date_possibs += cur_date_possibs
            cur_date_possibs = []
        
        dbl = False
        # Differentiate single from connected digits
        if width > 1.2 * prev_not_one_width:
            dbl = True
            roi_resized = cv2.resize(roi_pad, (38 ,28), interpolation=cv2.INTER_NEAREST)
            roi_cnn = np.expand_dims(roi_resized, axis=2)

            prob = model.predict_proba(np.array([roi_cnn]), verbose=0)
            dbl_nbr = np.argmax(prob)
                
            nbr_prob = prob[0]
            
            # copy the current possibs
            new_cur_date_possibs = copy.deepcopy(cur_date_possibs)
            
            if len(new_cur_date_possibs) == 0:
                new_cur_date_possibs = [[dbl_nbr]]
            else:
                for date_possib in new_cur_date_possibs:
                    date_possib.append(dbl_nbr)
                
        roi_resized = cv2.resize(roi_pad, (28 ,28), interpolation=cv2.INTER_NEAREST)
        
        # Deskew
        if should_deskew:
            roi_resized = deskew(roi_resized)
        
        roi_cnn = np.expand_dims(roi_resized, axis=2)

        prob = model_single.predict_proba(np.array([roi_cnn]), verbose=0)
        nbr = np.argmax(prob)
        nbr_prob = prob[0]
        
        if len(cur_date_possibs) == 0:
            cur_date_possibs = [[nbr]]
        else:
            for date_possib in cur_date_possibs:
                date_possib.append(nbr)
            
        if dbl:
            cur_date_possibs += new_cur_date_possibs
        
        prev_end_x = x_start
        prev_end_y = y_start

        # Mark the roi, label, and hierarchy
        cv2.rectangle(im_c, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 100, 255), 1)
        cv2.putText(im_c, str(int(nbr)), (rect[0], rect[1]), cv2.FONT_ITALIC, 0.4, (255, 0, 100), 1)
#         cv2.putText(im_c, str(hierarchy[0][i]) + str(int(nbr)), (rect[0], rect[1] - (250 - i*20)), cv2.FONT_ITALIC, 0.4, (randint(0,255), 0, 255), 1)
        
        if dbl:
            dbl_label = str(int(dbl_nbr))
            cv2.putText(im_c, dbl_label, (rect[0], rect[1] - 15), cv2.FONT_ITALIC, 0.3, (255, 0, 200), 1)
            
        probs.append(0)
        rois.append(roi_resized)
        
        if nbr != 1:
            prev_not_one_width = width
       
    # Append the final date possibility
    date_possibs += cur_date_possibs
    
    return im_c, rois, date_possibs, probs


# In[161]:

# images[13][0:790,0:1000] = [255, 255, 255]

im_labeled, rois, date_possibs, probs = extract_numbers(
#     images[16][650:780,800:],
#     images[3][700:880,490:1000],
    april_images[33][2000:,100:],
    thresh=80,
#     blue_thresh=True,
    separate_c=True,
#     blur=5,
    brightness_inc=50,
#     contrast_inc=1.5,
#     opening_shape=disk(5),
#     closing_shape=disk(5),
#     dilation_size=2,
#     erosion_size=3,
    should_deskew=True
)

plt.imshow(im_labeled)
plt.show()


# In[ ]:

plot_things(rois, range(len(rois)))


# In[ ]:

for d in date_possibs:
    print d


# In[ ]:



