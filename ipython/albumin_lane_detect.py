
# coding: utf-8

# In[1]:

import PyPDF2
from PIL import Image

import os
import sys
import warnings
import matplotlib
import numpy as np
from os import path

from skimage import data
from skimage import transform
from skimage.draw import circle
from skimage.util import img_as_float
from skimage.color import rgb2gray
from skimage.exposure import rescale_intensity

from skimage.feature import match_template # (only works for single match)?
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from skimage import data, img_as_float

from sklearn.cluster import KMeans

from matplotlib import pyplot as plt
warnings.filterwarnings("ignore")


# In[2]:

import os
import sys
# Allow to import local python modules here in Jupyter
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
from app.utils.preprocessing import *


# In[3]:

def load_april_2016_images():
    filenames = os.listdir('../data/april_2016_gels_renamed/')
    filenames = [x for x in filenames if 'pep1' in x]
    filenames = [x for x in filenames if not 'big' in x]
    
    imgs_april = []
    for filename in filenames:
        img = load_and_process_image('../data/april_2016_gels_renamed/' + filename)
        imgs_april.append(img)
        
    return imgs_april
    

def load_nov_2016_images():
    imgs_nov = []
    imgs_nov_idx = [1,6,12,21,41,42,51,52,56,83,84,89,90,96,97,106,123,131,136,152,153,156,157] # 7, 22
    
    for idx in imgs_nov_idx:
        filename = '../data/gels_nov_2016/Im{} - p. {}.png'.format(idx, idx)
        img = load_and_process_image(filename)
        imgs_nov.append(img)
        
    return imgs_nov


def load_and_process_image(filename):
    shape = (1276, 2100)

    cur_im = data.imread(filename, flatten=True)
    cur_im = img_as_float(cur_im)
    cur_im = rescale_intensity(cur_im)
    cur_im = rgb2gray(cur_im)

    cur_im = transform.resize(cur_im, output_shape=shape) # todo
    return cur_im


imgs_nov = load_nov_2016_images()
imgs_april = load_april_2016_images()
all_images = imgs_nov + imgs_april


# ### Grab Albumin roi

# In[4]:

alb = imgs_nov[0][307:398,460:507]


# In[ ]:

plt.imshow(alb)
plt.show()


# ### Find ROI in gel

# In[5]:

def find_matches(img, template):
    overlap_thresh = 50
    result = match_template(img, template)
    xy_max = np.unravel_index(np.argsort(result.ravel())[-500:], result.shape)
    
    zipped = zip(xy_max[0], xy_max[1])
    zipped_rev = np.flipud(zipped)
    found = np.zeros(img.shape)
    top_matches = [result[x, y] for x, y in zipped[0:100]]
    print('Mean top 100 match score: ', np.mean(top_matches))
    
    # Don't include same ROI twice
    xy_dedup = []
    for x, y in zipped_rev:
        # Maximum number of lanes
        if len(xy_dedup) >= 28: break
        # Minimum correlation
#         if result[x, y] < .8: break
            
        overlap = found[x : x + alb.shape[0], y : y + alb.shape[1]]
        if np.sum(overlap) < overlap_thresh:
            found[x : x + template.shape[0], y : y + template.shape[1]] = 1

            x_cen = x + int(template.shape[0] / 2)
            y_cen = y + int(template.shape[1] / 2)
            xy_dedup.append((x_cen, y_cen))
            
    return xy_dedup


# ### View marks

# In[6]:

def mark_match_rois(img, marker_points):
    for x, y in marker_points:
        rr, cc = circle(x, y, 5)
        img[rr, cc] = 1
    return img


# ### Extract lanes above Alb roi

# In[7]:

def extract_lanes_using_markers(img, markers):
    lanes = []
    i = 1
    # Weight X dimension higher than Y dimension
    markers_sorted = sorted(markers, key=lambda x: x[1] + 10*x[0])
    
    for x, y in markers_sorted:
        roi = img[x - 70 : x + 10, y - 10 : y + 10]
        lanes.append((roi, i))
        i += 1
    return lanes


# ### Plot lanes

# In[8]:

def plot_lanes(lanes, labels):
    count = len(lanes)
    plt.figure(figsize=(20, 20))
    
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    for i, lane in enumerate(lanes):
        cols = 40
        rows = int(count / cols) + 1
        ax = plt.subplot(rows, cols, 1 + i)
        
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_title(labels[i])

        plt.imshow(lane)


# In[9]:

# Choose nov, april, or all_images
to_analyze = all_images

all_markers = [find_matches(img, alb) for img in to_analyze]

marked = [mark_match_rois(img.copy(), markers) for img, markers in zip(to_analyze, all_markers)]

lanes_per_gel = [extract_lanes_using_markers(img, markers) for img, markers in zip(to_analyze, all_markers)]

# Flatten
all_lanes = [item for sublist in lanes_per_gel for item in sublist]
len(all_lanes)


# ### Min Dist b/w good & bad lanes

# In[10]:

# Gold std lanes
gld = all_lanes[0][0]
gld_bad = all_lanes[279][0]


# In[52]:

def group_lanes_per_date(all_lanes, labels, good_class):
    start = 0
    good_lanes_per_gel = []
    bad_lanes_per_gel = []
    
    for lanes in lanes_per_gel:
        current_labels = labels[start : start + len(lanes)]
        good_inds = np.array(np.array(current_labels) == good_class)
        bad_inds = np.array(np.array(current_labels) != good_class)

        good_lanes_per_gel.append(np.array(lanes)[good_inds])
        bad_lanes_per_gel.append(np.array(lanes)[bad_inds])
        start += len(lanes)
        
    return good_lanes_per_gel, bad_lanes_per_gel


# In[53]:

bad_dists = np.array([np.sum((x[0] - gld_bad)**2) for x in all_lanes])
good_dists = np.array([np.sum((x[0] - gld)**2) for x in all_lanes])
dist_labels = [0 if good_dists[i] < bad_dists[i] else 1 for i, dist in enumerate(bad_dists)]

bad_dists_inds = np.array(np.argsort(bad_dists))

# hard-code threshold
threshold = 19.0
bad_selected = np.array(all_lanes)[bad_dists < 19]
good_selected = np.array(all_lanes)[good_dists < 19]

# use min(good_dist, bad_dist)
good_lanes_per_gel, bad_lanes_per_gel = group_lanes_per_date(all_lanes, dist_labels, 0)

len(good_lanes_per_gel), len(good_lanes_per_gel[0])


# In[51]:

plot_lanes([x[0] for x in bad_lanes_per_gel[7]], ['a']*500)
plt.show()


# ### Cluster out bad lanes

# In[55]:

km = KMeans(n_clusters = 2)
km.fit(lanes_flat)

gld_label = np.argmin([np.sum((x.reshape(gld.shape) - gld)**2) for x in km.cluster_centers_])

labeled = [(img_and_lane_number, label) for img_and_lane_number, label in zip(lanes_flat, km.labels_)]
lanes_good = np.array(all_lanes)[km.labels_ == gld_label]
lanes_bad = np.array(all_lanes)[km.labels_ != gld_label]
    
good_lanes_per_gel, bad_lanes_per_gel = group_lanes_per_date(all_lanes, km.labels_, gld_label)
len(lanes_good), len(lanes_bad)


# ### IsoForest for bad lanes

# In[96]:

from sklearn.ensemble import IsolationForest
from scipy import stats


rng = np.random.RandomState(42)
n_samples = 200
outliers_fraction = 0.02
clusters_separation = [0, 1, 2]

iso = IsolationForest(
    max_samples=n_samples,
    contamination=outliers_fraction,
    random_state=rng)

iso.fit(lanes_flat)
scores_pred = iso.decision_function(lanes_flat)
threshold = stats.scoreatpercentile(scores_pred, 100 * outliers_fraction)
y_pred = iso.predict(lanes_flat)

labeled = [(img_and_lane_number, label) for img_and_lane_number, label in zip(lanes_flat, y_pred)]
lanes_good = np.array(all_lanes)[y_pred == 1]
lanes_bad = np.array(all_lanes)[y_pred != 1]


# In[97]:

len(lanes_good), len(lanes_bad)


# In[101]:

plot_lanes([x[0] for x in lanes_good[0:250]], ['b']*250)
plt.show()


# In[38]:

num = 15

# for num in range(len(all_markers)):
#     print num, len(bad_lanes_per_gel[num])

filenames = os.listdir('../data/april_2016_gels_renamed/')
filenames = [x for x in filenames if 'pep1' in x]
filenames = [x for x in filenames if not 'big' in x]

for i, filename in enumerate(filenames):
    bad_count = len(bad_lanes_per_gel[i])
    if bad_count > 0:
        print i, filename, bad_count


# ### Plot

# In[43]:

to_plot = 19

plt.imshow(marked[to_plot])

plot_lanes([x[0] for x in lanes_per_gel[to_plot]], ['a']*1000)
plot_lanes([x[0] for x in good_lanes_per_gel[to_plot]], ['g']*1000)
plot_lanes([x[0] for x in bad_lanes_per_gel[to_plot]], ['b']*1000)

plt.show()


# In[ ]:

plt.imshow(marked[4])
plt.show()


# ### Load Labels

# In[2]:

from datetime import datetime
from labels_collection import get_labels, get_dz_labels


april_2016_labels = get_labels(datetime(2016, 4, 1), datetime(2016, 4, 30))
april_2016_dz_labels = get_dz_labels(april_2016_labels)

nov_2016_labels = get_labels(datetime(2016, 11, 1), datetime(2016, 11, 30))
nov_2016_dz_labels = get_dz_labels(nov_2016_labels)


# In[11]:

import collections

od = collections.OrderedDict(sorted(nov_2016_dz_labels.items()))


# In[13]:

asdf = [1,6,12,21,41,42,51,52,56,83,84,89,90,96,97,106,123,131,136,152,153,156,157] # 7, 22

blah = zip([str(i + 1) + ' ' + str(val) for i, val in enumerate(asdf)], od.items())

blah


# In[57]:

X = [zip(calc_lane_means([z[0] for z in x]), [z[1] for z in x]) for x in good_lanes_per_gel]
X_flat = [z[0] for x in X for z in x]

print(len(imgs_nov), len(nov_2016_dz_labels.keys()))

y = []
for i, means in enumerate(X[0 : len(imgs_nov)]):
    date = nov_2016_dz_labels.keys()[i]
    for j in range(len(means)):
        if j in nov_2016_dz_labels[date]:
            y.append(1)
        else:
            y.append(0)

for i, means in enumerate(X[len(imgs_nov) : ]):
    date = april_2016_dz_labels.keys()[i]
    for j, k in means:
        if k in april_2016_dz_labels[date]:
            y.append(1)
        else:
            y.append(0)

len(X), len(X_flat), len(y)


# ### Classification

# In[223]:

from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split


x_train, x_test, y_train, y_test = train_test_split(X_flat, y)

clf = RandomForestClassifier()
clf.fit(x_train, y_train)
clf.score(x_test, y_test)


# In[ ]:

plt.figure(figsize=(20, 16))

X_means_df = pd.DataFrame(X_flat)

ctrl = X_means_df[np.array(y) == 0]
print len(ctrl), len(y)
plt.subplot(211)
ctrl.T.plot(alpha=.1, color='blue', ax=plt.gca(), legend=None, label='ctrl')

dz = X_means_df[np.array(y) == 1]
print len(dz), len(y)
plt.subplot(212)
dz.T.plot(alpha=.1, color='red', ax=plt.gca(), legend=None, label='dz')

plt.show()


# In[ ]:



