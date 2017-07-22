import matplotlib
matplotlib.use('Agg')

import os
from itertools import product
import scipy
from matplotlib import pyplot as plt
from skimage.morphology import square, disk

from digits_extractor import extract_numbers
from dates_searcher import find_dates


nov_imgs = [1,6,12,21,41,42,51,52,56,83,84,89,90,96,97,106,123,131,136,152,153,156,157]
nov_filenames = ['./data/gels_nov_2016/Im{} - p. {}.png'.format(i, i) for i in nov_imgs]
nov_images = [scipy.misc.imread(f) for f in nov_filenames]

# april_imgs = [f for f in os.listdir('./data/april_2016_gels_renamed') if not 'tore' in f]
# april_filenames = ['./data/april_2016_gels_renamed/{}'.format(f) for f in april_imgs]
# april_images = [scipy.misc.imread(f) for f in april_filenames]

# im = april_images[14][500:2500,240:2230]


param_grid = {
    'thresh': [20, 80],# 80],
    'blue_thresh': [False, True],
    'binary_roi': [False],
    'separate_c': [False],# False],
    'blur': [0, 2, 5],
    'brightness_inc': [0, 35],
    'contrast_inc': [0, 2],
    'opening_shape': [None, disk(3)],
    'closing_shape': [None, disk(2)],
    'dilation_size': [0, 2, 3],
    'erosion_size': [0, 3, 4],
    'should_deskew': [True, False]
}

# for i, im in enumerate(nov_images):
#     print("NEW FILE: ", nov_filenames[i])
#     run_extraction(im)


def run_extraction(im):
    params_set = [param_grid[key] for key in param_grid.keys()]
    params_combs = list(product(*params_set))
    params_combs_dicts = [dict(zip(param_grid.keys(), p)) for p in params_combs]

    for params in params_combs_dicts:
        try:
            print(params)
            im_labeled, rois, date_possibs, probs = extract_numbers(im, **params)

            plt.imshow(im_labeled)
            plt.show()

            # plot_things(rois, range(len(rois)))

            found_dates = find_dates(date_possibs)
            for d in found_dates:
                print "Found Date: " + d
        except Exception as e:
            print(e)
            continue


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
