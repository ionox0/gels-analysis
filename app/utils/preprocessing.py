import numpy as np
import pandas as pd
from skimage.transform import resize
from skimage.filters import threshold_otsu



def do_threshold(img):
    thresh = threshold_otsu(img)
#     thresh = threshold_adaptive(image, 15, 'mean')
    binary = img > thresh
    return np.array(binary)


def collapse_whitespace_margins(img, img_thresholded):
    vert_sum_img = img_thresholded.sum(axis=0)
    img_rolled = np.rollaxis(vert_sum_img, -1)
    max_intensity = np.max(vert_sum_img)
    margin_indices = np.where(img_rolled[2] == max_intensity)[0]

    mask_array = np.ones(img.shape[1], dtype=bool)
    mask_array[margin_indices] = False

    return np.array(img[:, mask_array, :])


def collapse_bottom_margins(img, img_thresholded):
    horiz_sum_img = img_thresholded.sum(axis=1)
    img_rolled = np.rollaxis(horiz_sum_img, -1)
    max_intensity = np.max(horiz_sum_img)

    mask_array = np.ones(img.shape[0], dtype=bool)
    bottom_margin_indices = []

    # todo - correct color channel?
    if img_rolled[0][0] != max_intensity:
        # Albumin reached bottom
        return img
    for i, k in enumerate(np.flip(img_rolled[0], axis=0)):
        if k == max_intensity:
            bottom_margin_indices.append(img_rolled[0].shape[0] - i - 1)
        else:
            break

    mask_array[bottom_margin_indices] = False
    return np.array(img[mask_array, :, :])


# Resize images
def resize_images(imgs, dim):
    imgs_resized = []
    for img in imgs:
        img_resized = resize(img, dim)
        imgs_resized.append(img_resized)
    return np.asarray(imgs_resized)


# Mean across horizontal axis
def calc_lane_means(lanes):
    lanes_means = []
    for lane in lanes:
        lane_mean = lane.mean(axis=1)
        if lane_mean.ndim == 3:
            # Just take the blue channels
            lanes_means.append(lane_mean[:, 2])
        else:
            lanes_means.append(lane_mean)

    return lanes_means


def smooth_lanes(X, N):
    smootheds = []
    for n in N:
        smoothed = [pd.rolling_mean(x, n)[n - 1:] for x in X]
        smootheds.append(smoothed)

    return smootheds
