import os
import sys
import scipy
import warnings
from itertools import product
import collections
from datetime import datetime

from skimage import transform
from skimage.draw import circle
from skimage.color import rgb2gray
from skimage.morphology import disk
from skimage.exposure import rescale_intensity
from skimage.feature import match_template # (only works for single match)?
from skimage import data, img_as_float

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split

from app.labels_collector.labels_collector import get_labels, get_dz_labels

from matplotlib import pyplot as plt

warnings.filterwarnings("ignore")





# Allow to import local python modules here in Jupyter
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from app.utils.preprocessing import *


ORIG_IMAGES = []

def load_april_2016_images():
    '''
    Read images data from data folder for April gels
    :return:
    '''
    filenames = os.listdir('./data/april_2016_gels_renamed/')
    filenames = [x for x in filenames if 'pep1' in x]
    filenames = [x for x in filenames if not 'big' in x]

    imgs_april = []
    for filename in filenames[0:20]:
        img = load_and_process_image('./data/april_2016_gels_renamed/' + filename)
        imgs_april.append(img)

    return imgs_april


def load_nov_2016_images():
    '''
    Read images data from data folder for November gels
    :return:
    '''
    imgs_nov = []
    imgs_nov_idx = [1, 6, 12, 21, 41, 42, 51, 52, 56, 83, 84, 89, 90, 96, 97, 106, 123, 131, 136, 152, 153, 156, 157]  # 7, 22

    nov_imgs = [f for f in os.listdir('./data/gels_nov_2016') if not 'tore' in f]
    nov_filenames = ['./data/gels_nov_2016/{}'.format(f) for f in nov_imgs]

    for idx in imgs_nov_idx[0:2]:
        filename = nov_filenames[idx]
        img = load_and_process_image(filename)
        imgs_nov.append(img)

    return imgs_nov


def load_and_process_image(filename):
    '''
    Read an image and resize to consistent sizing (hard coded for now)
    :param filename:
    :return:
    '''
    shape = (1276, 2100)

    ORIG_IMAGES.append(scipy.misc.imread(filename))

    cur_im = data.imread(filename, flatten=True)
    cur_im = img_as_float(cur_im)
    cur_im = rescale_intensity(cur_im)
    cur_im = rgb2gray(cur_im)

    cur_im = transform.resize(cur_im, output_shape=shape)  # todo
    return cur_im



def find_matches(img, template, alb):
    '''
    Find ROI in gel from albumin (or other) landmark
    :param img:
    :param template:
    :param alb:
    :return:
    '''
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

        overlap = found[x: x + alb.shape[0], y: y + alb.shape[1]]
        if np.sum(overlap) < overlap_thresh:
            found[x: x + template.shape[0], y: y + template.shape[1]] = 1

            x_cen = x + int(template.shape[0] / 2)
            y_cen = y + int(template.shape[1] / 2)
            xy_dedup.append((x_cen, y_cen))

    return xy_dedup


def mark_match_rois(img, marker_points):
    '''
    Mark matched regions for visualization purposes
    :param img:
    :param marker_points:
    :return:
    '''
    for x, y in marker_points:
        rr, cc = circle(x, y, 5)
        img[rr, cc] = 1
    return img


def extract_lanes_using_markers(img, markers):
    '''
    Extract lanes above roi (hard coded for now)
    :param img:
    :param markers:
    :return:
    '''
    lanes = []
    i = 1
    # Weight X dimension higher than Y dimension
    markers_sorted = sorted(markers, key=lambda x: x[1] + 10 * x[0])

    for x, y in markers_sorted:
        roi = img[x - 70: x + 10, y - 10: y + 10]
        lanes.append((roi, i))
        i += 1

    # Todo - should not be getting 0 widths...
    lanes = [x for x in lanes if x[0].shape[0] > 0]
    return lanes


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


def get_the_lanes(images, alb):
    '''
    Wrapper for finding and extracting lane ROIs
    :param images:
    :param alb:
    :return:
    '''
    # Choose nov, april, or all_images
    to_analyze = images

    all_markers = [find_matches(img, alb, alb) for img in to_analyze]

    # marked = [mark_match_rois(img.copy(), markers) for img, markers in zip(to_analyze, all_markers)]

    lanes_per_gel = [extract_lanes_using_markers(img, markers) for img, markers in zip(to_analyze, all_markers)]

    # Flatten
    all_lanes = [item for sublist in lanes_per_gel for item in sublist]
    len(all_lanes)

    return lanes_per_gel, all_lanes


def group_lanes_per_date(all_lanes, lanes_per_gel, labels, good_class):
    '''
    Min Dist b/w good & bad lanes (one of three possible methods of reducing bad matches)
    :param all_lanes:
    :param lanes_per_gel:
    :param labels:
    :param good_class:
    :return:
    '''
    start = 0
    good_lanes_per_gel = []
    bad_lanes_per_gel = []

    for lanes in lanes_per_gel:
        current_labels = labels[start: start + len(lanes)]
        good_inds = np.array(np.array(current_labels) == good_class)
        bad_inds = np.array(np.array(current_labels) != good_class)

        good_lanes_per_gel.append(np.array(lanes)[good_inds])
        bad_lanes_per_gel.append(np.array(lanes)[bad_inds])
        start += len(lanes)

    return good_lanes_per_gel, bad_lanes_per_gel


# Cluster out bad lanes
# plot_lanes([x[0] for x in bad_lanes_per_gel[7]], ['a']*500)
# plt.show()
#
# km = KMeans(n_clusters=2)
# km.fit(lanes_flat)
#
# gld_label = np.argmin([np.sum((x.reshape(gld.shape) - gld) ** 2) for x in km.cluster_centers_])
#
# labeled = [(img_and_lane_number, label) for img_and_lane_number, label in zip(lanes_flat, km.labels_)]
# lanes_good = np.array(all_lanes)[km.labels_ == gld_label]
# lanes_bad = np.array(all_lanes)[km.labels_ != gld_label]
#
# good_lanes_per_gel, bad_lanes_per_gel = group_lanes_per_date(all_lanes, km.labels_, gld_label)
# len(lanes_good), len(lanes_bad)


# Isoforest for bad lanes
# from sklearn.ensemble import IsolationForest
# from scipy import stats
#
#
# rng = np.random.RandomState(42)
# n_samples = 200
# outliers_fraction = 0.02
# clusters_separation = [0, 1, 2]
#
# iso = IsolationForest(
#     max_samples=n_samples,
#     contamination=outliers_fraction,
#     random_state=rng)
#
# iso.fit(lanes_flat)
# scores_pred = iso.decision_function(lanes_flat)
# threshold = stats.scoreatpercentile(scores_pred, 100 * outliers_fraction)
# y_pred = iso.predict(lanes_flat)
#
# labeled = [(img_and_lane_number, label) for img_and_lane_number, label in zip(lanes_flat, y_pred)]
# lanes_good = np.array(all_lanes)[y_pred == 1]
# lanes_bad = np.array(all_lanes)[y_pred != 1]


def retreive_labels(start_date, end_date):
    '''
    Load Labels from pre-existing Excel spreadsheet
    :param start_date:
    :param end_date:
    :return:
    '''
    labels = get_labels(start_date, end_date)
    dz_labels = get_dz_labels(labels)

    return dz_labels


# od = collections.OrderedDict(sorted(nov_2016_dz_labels.items()))
# asdf = [1,6,12,21,41,42,51,52,56,83,84,89,90,96,97,106,123,131,136,152,153,156,157] # 7, 22
#
# blah = zip([str(i + 1) + ' ' + str(val) for i, val in enumerate(asdf)], od.items())
#
# print blah


def build_labels(labels, date, lanes):
    '''
    Turn values from Excel spreadsheet into 0 (ctrl) or 1 (dz), and return in list
    :return:
    '''
    y = []
    for i in range(len(lanes)):
        if i in labels[date]:
            y.append(1)
        else:
            y.append(0)

    return y



### Classification
def auto_classify(X_flat, y):
    x_train, x_test, y_train, y_test = train_test_split(X_flat, y)

    clf = RandomForestClassifier()
    clf.fit(x_train, y_train)
    score = clf.score(x_test, y_test)
    print score

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



imgs_nov = load_nov_2016_images()
imgs_april = load_april_2016_images()
all_images = imgs_nov + imgs_april

### Grab Albumin roi
alb = imgs_nov[0][307:398, 460:507]
lanes_per_gel, all_lanes = get_the_lanes(all_images, alb)

print('Lanes per gel len: ', len(lanes_per_gel))
print('All lanes len: ', len(all_lanes))
print 'ORIG_IMAGES len: ', len(ORIG_IMAGES)


# Gold std lanes
gld = all_lanes[0][0]
# gld_bad = all_lanes[279][0]
gld_bad = all_lanes[179][0]

all_lanes_filt = all_lanes
bad_dists = np.array([np.sum((x[0] - gld_bad) ** 2) for x in all_lanes_filt])
good_dists = np.array([np.sum((x[0] - gld) ** 2) for x in all_lanes_filt])
dist_labels = [0 if good_dists[i] < bad_dists[i] else 1 for i, dist in enumerate(bad_dists)]

bad_dists_inds = np.array(np.argsort(bad_dists))

# hard-code threshold
threshold = 19.0
bad_selected = np.array(all_lanes_filt)[bad_dists < 19]
good_selected = np.array(all_lanes_filt)[good_dists < 19]

# use min(good_dist, bad_dist)
good_lanes_per_gel, bad_lanes_per_gel = group_lanes_per_date(all_lanes_filt, lanes_per_gel, dist_labels, 0)

print len(good_lanes_per_gel), len(good_lanes_per_gel[0])



param_grid = {
    'thresh': [80],# 20],
    'blue_thresh': [False],# True],
    'binary_roi': [True],# False],
    'separate_c': [False],# True],
    'blur': [0],# 3, 5],
    'brightness_inc': [35, 0],
    'contrast_inc': [0],# 2],
    'opening_shape': [disk(3), None],
    'closing_shape': [None, disk(2)],
    'dilation_size': [0, 2],# 3],
    'erosion_size': [0, 3],# 4],
    'should_deskew': [True],# False]
}

params_set = [param_grid[key] for key in param_grid.keys()]
params_combs = list(product(*params_set))
params_combs_dicts = [dict(zip(param_grid.keys(), p)) for p in params_combs]

from digits_extractor import extract_numbers
from dates_searcher import find_dates

found_boolv = []
found_dates = []
for im in ORIG_IMAGES:
    for params in params_combs_dicts:
        print params
        im_c, rois, date_possibs, probs = extract_numbers(im, **params)
        dates = find_dates(date_possibs)
        if len(dates):
            print "Found dates: ", dates
            found_dates.append(dates[0]) # Todo - just take first found date for now...
            found_boolv.append(1)
            break
    found_boolv.append(0)

print("found boolv: ", found_boolv)

labels = [retreive_labels(datetime.strptime(fd, '%Y-%m-%d'), datetime.strptime(fd, '%Y-%m-%d')) for fd in found_dates]

good_lanes_per_gel_filt = [x for i, x in enumerate(good_lanes_per_gel) if found_boolv[i] == 1]
X = [zip(calc_lane_means([z[0] for z in x]), [z[1] for z in x]) for x in good_lanes_per_gel_filt]

print labels, found_dates, len(X)

ys = [build_labels(labels[i], found_dates[i], x) for i, x in enumerate(X)]
y = [y for yx in ys for y in yx]

X_flat = [z[0] for x in X for z in x]
print len(X), len(X_flat), len(y)

print len(imgs_nov), len(labels)

auto_classify(X_flat, y)

