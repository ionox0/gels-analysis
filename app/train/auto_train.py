import PyPDF2
from PIL import Image
import warnings
import matplotlib
import numpy as np

from skimage import transform
from skimage.color import rgb2gray
from skimage.exposure import rescale_intensity

from skimage.feature import match_template
# import matplotlib.pyplot as plt
from skimage import data, img_as_float


from matplotlib import pyplot as plt
warnings.filterwarnings("ignore")



def load_images():
    imgs_blue = []
    imgs_blue_idx = [1, 6, 7, 12, 21, 22, 41, 42, 51, 52, 56, 83, 84, 89, 90, 96, 97, 106, 123, 131, 136, 152, 153, 156,
                     157]

    shape = (1276, 2100)

    for idx in imgs_blue_idx:
        cur_im = data.imread('../data/gels_nov_2016/Im{} - p. {}.png'.format(idx, idx), flatten=True)

        cur_im = img_as_float(cur_im)
        cur_im = rescale_intensity(cur_im)
        cur_im = rgb2gray(cur_im)
        cur_im = transform.resize(cur_im, output_shape=shape)  # todo

        imgs_blue.append(cur_im)

    return imgs_blue


def find_matches(img, template):
    result = match_template(img, template)
    xy_max = np.unravel_index(np.argsort(result.ravel())[-100:], result.shape)
    found = np.zeros(img.shape)

    # Don't include same ROI twice
    xy_dedup = []
    for x, y in zip(xy_max[0], xy_max[1]):
        overlap = found[x: x + template.shape[0], y: y + template.shape[1]]
        if not overlap.any():
            found[x: x + template.shape[0], y: y + template.shape[1]] = 1

            x_cen = x + int(template.shape[0] / 2)
            y_cen = y + int(template.shape[1] / 2)
            xy_dedup.append((x_cen, y_cen))

    return xy_dedup


def extract_lanes_from_marker(img, markers):
    lanes = []
    for x, y in markers:
        lanes.append(img[x - 70 : x, y - 10 : y + 10])
    return lanes


def plot_lanes(lanes, titles):
    count = len(lanes)
    plt.figure(figsize=(20, 20))
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    for i, lane in enumerate(lanes):
        cols = 40
        rows = int(count / cols) + 1
        plt.subplot(rows, cols, 1 + i)
        plt.imshow(lane)


def alb_auto_extract(alb_roi):
    imgs_blue = load_images()

    alb = imgs_blue[0][357:378, 460:507]

    all_markers = [find_matches(img, alb) for img in imgs_blue]
    l = [extract_lanes_from_marker(img, markers) for img, markers in zip(imgs_blue, all_markers)]
    lanes = [item for sublist in l for item in sublist]
    print(len(lanes))

