import matplotlib
matplotlib.use('agg')

import re
import dateutil.parser

import os
import scipy
import logging
import multiprocessing
from itertools import product
from skimage.morphology import disk
from matplotlib import pyplot as plt


logging.basicConfig(filename='test_extractor.log', level=logging.DEBUG,
                    format='%(asctime)s:%(levelname)s:%(message)s')


nov_imgs = [f for f in os.listdir('./data/gels_nov_2016') if not 'tore' in f]
nov_filenames = ['./data/gels_nov_2016/{}'.format(f) for f in nov_imgs]
# Go back to this if errors:
nov_images = [scipy.misc.imread(f) for f in nov_filenames]
# nov_images = [cv2.imread(f) for f in nov_filenames]

# april_imgs = [f for f in os.listdir('./data/april_2016_gels_renamed') if not 'tore' in f]
# april_filenames = ['./data/april_2016_gels_renamed/{}'.format(f) for f in april_imgs]
# april_images = [scipy.misc.imread(f) for f in april_filenames]
# im = april_images[14][500:2500,240:2230]



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


def run_extract_with_params(im, params_combs_dicts, date_parsed):
    from digits_extractor import extract_numbers
    from dates_searcher import find_dates

    for params in params_combs_dicts:
        try:
            im_labeled, rois, date_possibs, probs = extract_numbers(im, **params)

            found_dates = find_dates(date_possibs)
            if check_date_answer(found_dates, date_parsed):
                logging.info('Correct params: ', params)
                return True

        except Exception as e:
            logging.error('ERROR: ', e)
            continue

    return False


def check_date_answer(found_dates, date_parsed):
    for d in found_dates:
        logging.info("Found Date: " + d)

        if d == date_parsed.strftime('%Y-%m-%d'):
            logging.info('Correct Date Found')
            return True
    return False


def extract_one(i):
    param_grid = {
        'thresh': [80, 20],
        'blue_thresh': [False, True],
        'binary_roi': [True, False],
        'separate_c': [False, True],
        'blur': [0, 3, 5],
        'brightness_inc': [35, 0],
        'contrast_inc': [0, 2],
        'opening_shape': [disk(3), None],
        'closing_shape': [None, disk(2)],
        'dilation_size': [0, 2, 3],
        'erosion_size': [0, 3, 4],
        'should_deskew': [True, False]
    }

    im = nov_images[i].copy()
    filename = nov_filenames[i]
    logging.info('Extracting dates from file: ', filename)

    date = re.search(r'(\d\d-\d\d?-\d\d?)', filename).groups()[0]
    date_parsed = dateutil.parser.parse(date)

    params_set = [param_grid[key] for key in param_grid.keys()]
    params_combs = list(product(*params_set))
    params_combs_dicts = [dict(zip(param_grid.keys(), p)) for p in params_combs]

    if run_extract_with_params(im, params_combs_dicts, date_parsed):
        return True

    return False



if __name__ == "__main__":
    pool = multiprocessing.Pool(4)
    result = pool.map(extract_one, range(len(nov_images)))

    logging.info('Finished')

    for value in result:
        logging.info(value)

