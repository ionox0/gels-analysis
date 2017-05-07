import os
# import shutil
import itertools
import logging
import sys
import matplotlib
import numpy as np
# from matplotlib import pyplot as plt

from skimage import data, filters, transform
from skimage.filters import threshold_otsu


sys.path.insert(0, '/Users/ianjohnson/.virtualenvs/cv/lib/python2.7/site-packages/')
logger = logging.getLogger(__name__)

DELIMITER = '\n' + '*' * 30 + ' '


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
        jpgfile = open("jpg%d.jpg" % njpg, "wb")
        jpgfile.write(jpg)
        jpgfile.close()

        njpg += 1
        i = iend


def get_raw_images_in_dir(dir):
    def jpg(filename):
        return 'jpg' in filename

    raw_imgs = []
    for img_file in filter(jpg, os.listdir(dir)):
        img = data.imread(dir + '/' + img_file, 0)
        # todo - bad
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


def do_threshold(img):
    thresh = threshold_otsu(img)
    binary = img > thresh
    return binary


def isolate_lanes(img):
    vert_sum_img = calc_img_vertical_sum(img)
    img_rolled = np.rollaxis(vert_sum_img, -1)
    max_intensity = np.max(vert_sum_img)
    separator_indices = np.where(img_rolled[2] == max_intensity)[0]

    lanes = []
    coords = []
    for i, val in enumerate(separator_indices):
        if i == 0: continue
        if separator_indices[i] != separator_indices[i - 1] + 1:
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


def analyze_gel(filename, rois, danger_zone, threshold):
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