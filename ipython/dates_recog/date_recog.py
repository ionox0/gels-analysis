
# coding: utf-8

# In[1]:

import cv2
import scipy
import pprint
import numpy as np
from sklearn.svm import SVC
from sklearn import datasets
from collections import Counter
from skimage.feature import hog
from sklearn import preprocessing
from skimage import data
from sklearn.externals import joblib

import keras
from matplotlib import pyplot as plt
get_ipython().magic(u'matplotlib auto')

pp = pprint.PrettyPrinter(indent=4)
n = 12
l = 256
np.random.seed(1)


# In[2]:

clf, pp = joblib.load("digits_cls.pkl")
model = keras.models.load_model('./digits_cnn')

imgs = [1,6,12,21,41,42,51,52,56,83,84,89,90,96,97,106,123,131,136,152,153,156,157] #7, 22


# ### HOG Features

# In[456]:

def calc_hog_feats(rect, roi):
    # Make the rectangular region around the digit
    leng = int(rect[3] * 1.6)
    pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
    pt2 = int(rect[0] + rect[2] // 2 - leng // 2)

    # todo
    if roi.shape[0] == 0 or roi.shape[1] == 0: return False

    # Resize the image
    roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
    roi = cv2.dilate(roi, (3, 3))

    # Calculate the HOG features
    roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
    roi_hog_fd = pp.transform(np.array([roi_hog_fd], 'float64'))
    
    return roi_hog_fd


# ### Sorting ROIs

# In[520]:

from collections import deque

def sort_rects(rects_ctrs):
    rects_sort = sorted(rects_ctrs, key=lambda r: r[0][0])
    
    result = []
    for rect in rects_sort:
        # Filter to only rects with overlap of > 1/3 of target height
        rects_sort_filt = filter(lambda x: calc_overlap(x[0], rect[0]) > (x[0][3] / 3.0) and abs(x[0][3] - rect[0][3]) < (x[0][3]), rects_sort)
        rects_sort_filt = sorted(rects_sort_filt, key=lambda r: r[0][0])
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


# ### Color thresholding

# In[459]:

image = scipy.misc.imread(filenames[0])
cv2.threshold(image, 180, 255, cv2.THRESH_BINARY)

# plt.imshow(image[800:802,600:602])

# [820:822,655:657] # blue roi
# [800:802,600:602] # black roi

# plt.imshow(image[580:880,440:980])
#plt.show()


# In[384]:

image[800:802,600:602]


# In[460]:

image = scipy.misc.imread(filenames[0])

boundaries = {
#     ([30, 15, 20], [50, 40, 60]), # black ink
    'black': ([0, 0, 0], [190, 190, 190]), # black ink
    'blue': ([50, 31, 4], [85, 120, 180]), # blue dye
}

colors = []
for bound in boundaries.keys():
    lower, upper = boundaries[bound]
    lower = np.array(lower, dtype = "uint8")
    upper = np.array(upper, dtype = "uint8")

    mask = cv2.inRange(image, lower, upper)
    output = cv2.bitwise_and(image, image, mask = mask)
    colors.append(output)

plt.imshow(np.hstack(colors))


# ### Numbers xtract fn

# In[525]:

import pdb

def extract_numbers(fname, thresh=80, blur=True):
    result = []
    
    im = scipy.misc.imread(fname)#[580:880,440:980]
    im_one = im.astype(np.uint8)[:,:,2]

    # Convert to grayscale and apply Gaussian filtering
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im_gray = (255 - im_one)
    if blur:
        im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)

    # Threshold the image
    ret, im_th_all_colors = cv2.threshold(im_gray, thresh, 255, cv2.THRESH_BINARY)
    # Or try with black threshold
    lower, upper = boundaries['black']
    lower = np.array(lower, dtype = "uint8")
    upper = np.array(upper, dtype = "uint8")
    mask = cv2.inRange(im, lower, upper)
    im_th = cv2.bitwise_and(im, im, mask = mask).astype(np.uint8)
    im_th = cv2.cvtColor(im_th, cv2.COLOR_BGR2GRAY)

    # Find contours in the image
    ctrs = cv2.findContours(im_th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get contour bounding boxes
    rects = [cv2.boundingRect(ctr) for ctr in ctrs[1]]
    
    # Sorted order of rois
    sorted_rects, sorted_ctrs = sort_rects(zip(rects, ctrs[1]))

    probs = []
    date_possibs = []
    cur_date_possib = []
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

        im_th_all_colors = im_th_all_colors.astype(np.float64)
        im_roi = im_th_all_colors[y_start : y_start + height, x_start : x_start + width]

        mask = np.zeros((height, width)).astype(np.uint8)
        mask = cv2.drawContours(mask, sorted_ctrs, i, (255, 255, 255), cv2.FILLED, offset=(-x_start, -y_start))
        roi = cv2.bitwise_and(mask.astype(np.uint8), im_roi.astype(np.uint8)).astype(np.uint8)

        # CNN
        if height < width:
            padding = int((width - height) / 2.0)
            roi_pad = np.pad(roi, (padding, padding), 'constant', constant_values=(0, 0))
        elif width < height:
            padding = int((height - width) / 2.0)
            roi_pad = np.pad(roi, (padding, padding), 'constant', constant_values=(0, 0))
        else:
            roi_pad = roi
        roi_pad = np.pad(roi_pad, (5, 5), 'constant', constant_values=(0, 0))

        roi_resized = cv2.resize(roi_pad, (28,28), interpolation=cv2.INTER_NEAREST)
        
        roi_cnn = np.expand_dims(roi_resized, axis=2)
        prob = model.predict_proba(np.array([roi_cnn]), verbose=0)
        nbr = np.argmax(prob)
        nbr_prob = prob[0]

        # SVM w. HOG feats
#         roi_hog_fd = calc_hog_feats(rect, roi)
#         if not hasattr(roi_hog_fd, 'shape'): continue
#         nbr = clf.predict(roi_hog_fd)
#         # Dummy prob
#         nbr_prob = 0

        if x_start - prev_end_x < 80 and y_start - prev_end_y < 80:
            cur_date_possib.append(nbr)
        else:
            date_possibs.append(cur_date_possib)
            cur_date_possib = [nbr]

        prev_end_x = x_start
        prev_end_y = y_start

        cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 100, 255), 1)
        cv2.putText(im, str(int(nbr)), (rect[0], rect[1]), cv2.FONT_ITALIC, 0.4, (0, 0, 255), 1)
        
        # Only add if we are confident
#         class_prob = np.max(prob)
#         if class_prob > .9:
#         probs.append(class_prob)

        if nbr == 9:
            continue
#             pdb.set_trace()
            
        probs.append(0)
        result.append((roi_resized, nbr))   

    date_possibs.append(cur_date_possib)
    
    plt.imshow(im)
    plt.show()
    
    return result, date_possibs, probs


# In[527]:

filenames = ['../data/gels_nov_2016/Im{} - p. {}.png'.format(i, i) for i in imgs][8:9]
results = [extract_numbers(f) for f in filenames]
res = [x[0] for x in results]
all_date_possibs = [x[1] for x in results]
probs = [x[2] for x in results]


# In[480]:

plt.imshow(scipy.misc.imread(filenames[0]))
plt.show()


# ### Plot res

# In[464]:

def plot_things(things, labels):
    count = len(things)
    plt.figure(figsize=(50, 50))
    
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    for i, thing in enumerate(things):
        cols = 15
        rows = int(count / cols) + 1
        ax = plt.subplot(rows, cols, 1 + i)
        
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_title(labels[i])

        plt.imshow(thing)


# In[512]:

rois_flat = [x[0] for sub in res for x in sub]
nbrs_flat = [x[1] for sub in res for x in sub]
probs_flat = [x for sub in probs for x in sub]

titles = ['{} {:.0%}'.format(x, y) for x, y in zip(nbrs_flat, probs_flat)]

i = 50
s = slice(i, i + 1000)
plot_things(rois_flat[s], titles[s])


# ### Search for date

# In[528]:

import re
from datetime import datetime

date_regex = r'([01]?[0-9])1?([0-3]?[0-9])1?([0-9]{2,4})'

# Filter matches to ones with b/w 6 and 8 characters
all_d = [filter(lambda x: len(x) > 4 and len(x) < 11, date_possibs) for date_possibs in all_date_possibs]

# Concatenate digit lists into strings
all_d_cat = [''.join([str(v) for v in x]) for d in all_d for x in d]

# Filter out matches with date_regex
all_d_filt = [filter(lambda x: re.match(date_regex, x), d) for d in all_d_cat]

# Extract date groups
all_d_match = [map(lambda x: re.search(date_regex, x).groups(), d) for d in all_d_filt]

# Take 0th elem
the_d = [d_match[0] if len(d_match) else None for d_match in all_d_match]

# Date objs
dates = [datetime(int('20' + d[2]), int(d[0]), int(d[1])) if d else None for d in the_d]

# Formatted
dates = [d.strftime('%Y-%m-%d') if d else None for d in dates]

dates


# In[ ]:



