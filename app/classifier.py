import os
# import shutil
import itertools
import logging
import sys
import matplotlib
import numpy as np
# from matplotlib import pyplot as plt

from skimage import data, filters, transform
from sklearn.externals import joblib

from utils.preprocessing import *
from utils.pdf_extractor import extract_images_from_pdf


sys.path.insert(0, '/Users/ianjohnson/.virtualenvs/cv/lib/python2.7/site-packages/')
logger = logging.getLogger(__name__)

DELIMITER = '\n' + '*' * 30 + ' '


def get_raw_images_in_dir(dir, rotate=True):
    def jpg(filename):
        return 'jpg' in filename

    raw_imgs = []
    for img_file in filter(jpg, os.listdir(dir)):
        img = data.imread(dir + '/' + img_file, 0)
        # todo - bad
        if rotate:
            img = transform.rotate(img, 90.0)
        raw_imgs.append(img)

    return raw_imgs


# Mean across horizontal axis
def calc_lane_means(lanes):
    lanes_means = []
    for lane in lanes:
        lane_mean = lane.mean(axis=1)
        lanes_means.append(lane_mean)
        # Just take the blue channels
        # lanes_means.append(lane_mean[:, 2])

    return lanes_means


# Detect Danger Lanes
def find_danger_lanes(lanes, danger_zone, threshold):
    y_means_lanes = calc_lane_means(lanes)

    y_min = danger_zone['y_min']
    y_max = danger_zone['y_max']

    indices = [np.any(x[y_min : y_max] < threshold) for x in y_means_lanes]
    maybe_dz = filter(lambda x: np.any(x[y_min:y_max] < threshold), y_means_lanes)

    dz_lanes = list(itertools.compress(maybe_dz, indices))
    logger.info(DELIMITER + 'Danger lanes detected: ' + str(len(dz_lanes)))

    indices = [i for i, x in enumerate(indices) if x]
    return indices, dz_lanes


def extract_roi(image, roi_metadata):
    roi = image[
          roi_metadata['y_start']: roi_metadata['y_end'],
          roi_metadata['x_start']: roi_metadata['x_end']
          ]
    return roi


def calc_img_vertical_sum(img):
    vert_sum = img.sum(axis=0)
    return vert_sum


def isolate_lanes(img, img_orig=None):
    # todo - img_orig
    vert_sum_img = calc_img_vertical_sum(img)
    img_rolled = np.rollaxis(vert_sum_img, -1)
    max_intensity = np.max(vert_sum_img)
    separator_indices = np.where(img_rolled[2] == max_intensity)[0]

    lanes = []
    coords = []
    for i, val in enumerate(separator_indices):
        if i == 0: continue
        if separator_indices[i] != separator_indices[i - 1] + 1:
            if img_orig != None:
                lanes.append(img_orig[:, separator_indices[i - 1]: val])
            else:
                lanes.append(img[:, separator_indices[i - 1]: val])
            coords.append((separator_indices[i - 1], val))
    return lanes, coords


def label_img(img, danger_coords):
    # todo - better struct

    for danger_coord in danger_coords:
        img[
            danger_coord[1]['y_start'] - 50 : danger_coord[1]['y_start'],
            danger_coord[1]['x_start'] + danger_coord[0][0] : danger_coord[1]['x_start'] + danger_coord[0][1]
        ] = [1,0,0]

    return img


def clear_data():
    folder = './uploaded_data'
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
                # elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)


def manual_classify_gel(filename, rois, danger_zone, threshold):
    if '.pdf' in filename:
        extract_images_from_pdf(filename)

    dir = './uploaded_data'
    imgs = get_raw_images_in_dir(dir)

    # Techs can only upload one gel per PDF for now
    img = imgs[0]

    all_lanes = []
    all_coords = []
    for roi_metadata in rois:
        img_roi = extract_roi(img, roi_metadata)
        roi_thresholded = do_threshold(img_roi).astype(np.uint16)
        lanes, coords = isolate_lanes(roi_thresholded)

        all_lanes += lanes

        # todo - weird struct
        # (x_start, x_end), ({x_start, x_end, y_start, y_end})
        coords_x_y = zip(coords, itertools.repeat(roi_metadata))
        all_coords += coords_x_y

    danger_indices, danger_lanes = find_danger_lanes(all_lanes, danger_zone, threshold)

    danger_coords = [all_coords[idx] for idx in danger_indices]
    labeled_image = label_img(img, danger_coords)

    result_filename = 'labeled_image.jpg'
    matplotlib.image.imsave(result_filename, labeled_image)

    clear_data()
    return result_filename


def auto_classify_gel(filename, rois):
    dir = './uploaded_data'
    img = get_raw_images_in_dir(dir, rotate=False)[0]

    all_lanes = []
    all_coords = []
    for roi_metadata in rois:
        img_roi = extract_roi(img, roi_metadata)
        roi_thresholded = do_threshold(img_roi).astype(np.uint16)
        lanes, coords = isolate_lanes(roi_thresholded, img_roi)

        all_lanes += lanes

        # todo - weird struct
        # (x_start, x_end), ({x_start, x_end, y_start, y_end})
        coords_x_y = zip(coords, itertools.repeat(roi_metadata))
        all_coords += coords_x_y

    # X = np.array(all_lanes)
    # print X.shape

    X = all_lanes

    X_threshold = [do_threshold(x).astype(np.uint16) for x in X]
    # X_threshold = np.array(X_threshold)
    print len(X_threshold)

    X_collapsed = [collapse_whitespace_margins(x, z) for x, z in zip(X, X_threshold)]
    X_collapsed_vert = [collapse_bottom_margins(x, z) for x, z in zip(X_collapsed, X_threshold)]
    X_resized = resize_images(X_collapsed_vert, dim = (60, 233))
    print len(X_resized)

    X_means = np.array(calc_lane_means(X_resized))
    X_means = X_means[:,:,2]
    print len(X_means), X_means[0].shape

    clf = joblib.load('trained_classifier.pkl')
    preds = clf.predict(X_means)

    clear_data()
    return preds
