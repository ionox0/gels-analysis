
# coding: utf-8

# In[2]:

import cv2
import scipy
import pprint
import numpy as np
from collections import deque
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

imgs = [1,6,12,21,41,42,51,52,56,83,84,89,90,96,97,106,123,131,136,152,153,156,157] #7, 22
filenames = ['../data/gels_nov_2016/Im{} - p. {}.png'.format(i, i) for i in imgs]


# In[4]:

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


# In[325]:

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

def extract_numbers(im, thresh=80, blur=True):
    result = []
    
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
        
        print("MASK", mask)
        print("CONTOURS", sorted_ctrs)
        print("i", i)
        
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


# In[326]:

im = scipy.misc.imread(filenames[8])#[580:880,440:980]   
roi = im[790:880,520:750]
imgs = [roi]

results = [extract_numbers(i) for i in imgs]

res = [x[0] for x in results]
all_date_possibs = [x[1] for x in results]
probs = [x[2] for x in results]


# In[8]:

# https://gist.github.com/nvictus/66627b580c13068589957d6ab0919e66
def rlencode(x, dropna=False):
    where = np.flatnonzero
    x = np.asarray(x)
    n = len(x)
    if n == 0:
        return (np.array([], dtype=int), 
                np.array([], dtype=int), 
                np.array([], dtype=x.dtype))

    starts = np.r_[0, where(~np.isclose(x[1:], x[:-1], equal_nan=True)) + 1]
    lengths = np.diff(np.r_[starts, n])
    values = x[starts]
    
    if dropna:
        mask = ~np.isnan(values)
        starts, lengths, values = starts[mask], lengths[mask], values[mask]
    
    return starts, lengths, values


# In[9]:

# Alg from paper:
# http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=1363926

thresh=150 # seems to be good for digits connected to gel edge...
ima = roi.astype(np.uint8)[:,:,2]
ret, ima = cv2.threshold(ima, 150, 255, cv2.THRESH_BINARY)

# estimate median stroke width lambda
rle = rlencode(ima.ravel())
lam = np.median(rle[1])


# In[351]:

def detect_connected_digits(image):
    prob = model.predict_proba(image, verbose=0)
    
    if np.any(prob > .7):
        print('single digit: ', prob)
        return False
    else:
        return True

def smooth_contours(contours):
    pass

def get_corner_points(image):
    pass

def filter_corner_points_to_segmentation_points(image):
    pass


# In[11]:

def projection_segmentation_points(image):
    pass

def get_segmentations(image, points):
    pass

def most_probable_segmentation(segmentations):
    pass


# In[356]:

new_roi = ima[30:65,125:175]


# In[419]:

inverted = np.invert(opened)
kernel = np.ones((5, 5), np.uint8)
# this is ugly but is best way found to connect chessboard corners
# todo - cv2.chessboardCorners...?
x = skimage.morphology.binary_closing(inverted, kernel)
plt.imshow(x)


# In[420]:

im2, contours, hierarchy = cv2.findContours(x.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)   

new = np.zeros((50, 50, 3)).astype(np.uint8)
mask = cv2.drawContours(new, contours, 0, (255, 255, 255), cv2.FILLED, offset=(-1, -1))


# In[421]:

def find_leftmost_contour_point(contour):
    x_min = 9999
    y_min = 9999
    for point in contour:
        # todo - wtf are these points doubly nested?
        point = point[0]
        if point[0] < x_min or (point[0] == x_min and point[1] < y_min):
            x_min = point[0]
            y_min = point[1]
    return [x_min, y_min]

def find_rightmost_contour_point(contour):
    x_max = 0
    y_min = 0
    for point in contour:
        point = point[0]
        if point[0] > x_max or (point[0] == x_max and point[1] < y_min):
            x_max = point[0]
            y_min = point[1]
    return [x_max, y_min]

start = find_leftmost_contour_point(contours[0])
end = find_rightmost_contour_point(contours[0])


# In[422]:

def trace_half_contour_ltr(contour, start, end, reverse=False):
    """
    Returns either the bottom or top contour of `contour` from the
    `start` to `end` points and b/t based on reverse param
    """
    if reverse:
        contour = contour[::-1]
        
    half_contour = []
    started = False
    i = 1
    while True:
        try:
            point = contour[i]
            point = point[0]
            if point[0] == start[0] and point[1] == start[1]:
                started = True
            if started:
                half_contour.append([point.tolist()])
            if point[0] == end[0] and point[1] == end[1] and len(half_contour):
                return np.array(half_contour).astype(np.uint8)
            i = i + 1
        except:
            i = 0
        
bottom_contour = trace_half_contour_ltr(contours[0], start, end)       
top_contour = trace_half_contour_ltr(contours[0], start, end, reverse=True)


# In[423]:

zeros = np.zeros((50,50), np.int32)
topctrlist = top_contour.astype(np.int32)

def draw_contour(points):
    frame = np.zeros((50, 50), np.uint8)
    prev = points[0][0]
    for p in points[1:]:
        p = p[0]
        frame = cv2.line(frame, (prev[0], prev[1]), (p[0], p[1]), (255, 255, 255), 1)
        prev = p
    return frame

plt.imshow(draw_contour(bottom_contour))


# In[ ]:



