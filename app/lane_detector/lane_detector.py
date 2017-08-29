import warnings

from scipy import stats
from skimage.draw import circle
from skimage.feature import match_template # (only works for single match)?
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

from app.utils.preprocessing import *

warnings.filterwarnings("ignore")


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


def cluster_out_bad_lanes(bad_lanes_per_gel, lanes_flat, gld, all_lanes):
    plot_lanes([x[0] for x in bad_lanes_per_gel[7]], ['a']*500)
    plt.show()

    km = KMeans(n_clusters=2)
    km.fit(lanes_flat)

    gld_label = np.argmin([np.sum((x.reshape(gld.shape) - gld) ** 2) for x in km.cluster_centers_])

    # labeled = [(img_and_lane_number, label) for img_and_lane_number, label in zip(lanes_flat, km.labels_)]
    lanes_good = np.array(all_lanes)[km.labels_ == gld_label]
    lanes_bad = np.array(all_lanes)[km.labels_ != gld_label]

    good_lanes_per_gel, bad_lanes_per_gel = group_lanes_per_date(all_lanes, km.labels_, gld_label)
    print len(lanes_good), len(lanes_bad), len(good_lanes_per_gel), len(bad_lanes_per_gel)

    return lanes_good, lanes_bad, good_lanes_per_gel, bad_lanes_per_gel


def isoforest_out_bad_lanes(lanes_flat, all_lanes):
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


def good_bad_euclid(gld, gld_bad, all_lanes_filt):
    bad_dists = np.array([np.sum((x[0] - gld_bad) ** 2) for x in all_lanes_filt])
    good_dists = np.array([np.sum((x[0] - gld) ** 2) for x in all_lanes_filt])
    dist_labels = [0 if good_dists[i] < bad_dists[i] else 1 for i, dist in enumerate(bad_dists)]

    bad_dists_inds = np.array(np.argsort(bad_dists))

    # Hard-code threshold
    threshold = 19.0
    bad_selected = np.array(all_lanes_filt)[bad_dists < threshold]
    good_selected = np.array(all_lanes_filt)[good_dists < threshold]

    return dist_labels