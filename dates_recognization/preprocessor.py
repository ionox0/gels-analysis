import cv2

print(cv2.__file__)

import keras
import numpy as np
from skimage.feature import hog
from skimage.morphology import erosion, dilation
from skimage.morphology import disk
from sklearn.externals import joblib



# model_differentiator_svm = joblib.load('./models/dbl_digits_differentiator_balanced_hog.pkl')
model_differentiator_cnn = keras.models.load_model('./models/double_digits_differentiator')
dbl_single_feats_pp = joblib.load('./models/dbl_single_feats_balanced_pp.pkl')


def thresh_img(im, thresh, blue_thresh=False):
    if blue_thresh:
        # RGB --> BGR (openCV style)
        im_bgr = im.copy()[:, :, ::-1]
        hsv = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2HSV)
        # Define range of blue color in HSV
        lower_blue = np.array([110, 50, 50])
        upper_blue = np.array([130, 255, 255])
        # Threshold the HSV image to get only blue colors
        im_th = cv2.inRange(hsv, lower_blue, upper_blue)
        im_th = cv2.bitwise_and(im_bgr, im_bgr, mask=im_th)
        im_th = cv2.cvtColor(im_th, cv2.COLOR_BGR2GRAY)
    else:
        ret, im_th = cv2.threshold(im, thresh, 255, cv2.THRESH_BINARY)
        im_th = im_th.astype(np.uint8)
    return im_th


def im_preprocessing(im_gray, blur, brightness_inc, contrast_inc, dilation_size, erosion_size):
    # Gaussian blur
    if blur:
        im_gray = cv2.medianBlur(im_gray, blur, blur)

    # Brightness
    if brightness_inc:
        im_expanded = im_gray.astype(np.uint64) + brightness_inc
        im_gray = np.clip(im_expanded, 0, 255).astype(np.uint8)
    # Contrast
    if contrast_inc:
        im_expanded = im_gray.astype(np.uint64) * contrast_inc
        im_gray = np.clip(im_expanded, 0, 255).astype(np.uint8)

    # Dilation
    if dilation_size:
        selem = disk(dilation_size)
        im_gray = dilation(im_gray, selem)
    # Erosion
    if erosion_size:
        selem = disk(erosion_size)
        im_gray = erosion(im_gray, selem)

    return im_gray


def find_rects_ctrs(im_th):
    # Find Contours
    _, ctrs, hierarchy = cv2.findContours(im_th, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    # Filter based on hierarchy
    ctrs = [c for i, c in enumerate(ctrs) if hierarchy[0][i][-1] == -1]
    hierarchy = [h for i, h in enumerate(hierarchy[0]) if hierarchy[0][i][-1] == -1]
    # Get Contour bounding boxes
    rects = [cv2.boundingRect(ctr) for ctr in ctrs]
    # Remove small rects
    ctrs = [c for i, c in enumerate(ctrs) if rects[i][3] > 10]
    rects = [r for r in rects if r[3] > 10]
    return rects, ctrs


def check_dbl(roi):
    #     hog_feats = calc_hog_feats(roi)
    #     is_dbl = model_differentiator_svm.predict(hog_feats)
    is_dbl = model_differentiator_cnn.predict_classes(np.array([np.expand_dims(roi, 2)]), batch_size=1, verbose=0)
    return is_dbl


def calc_hog_feats(roi):
    fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
    # Normalize the features
    hog_features = dbl_single_feats_pp.transform([fd])
    return hog_features


def split_dbls(rects, ctrs, im_gray):
    split_rects = []
    split_ctrs = []
    singles = []
    dbls = []
    for i, rect in enumerate(rects):
        roi = draw_roi(rect, i, ctrs, im_gray, None)
        is_dbl = check_dbl(roi)
        if is_dbl:
            new_rects, new_ctrs = separate_connected(rect, ctrs[i], im_gray)
            split_rects += new_rects
            split_ctrs += new_ctrs
            dbls.append(roi)
        else:
            split_rects.append(rect)
            split_ctrs.append(ctrs[i])
            singles.append(roi)

    return split_rects, split_ctrs


def separate_connected(rect, ctr, im_gray):
    x_start = rect[0]
    y_start = rect[1]
    width = rect[2]
    height = rect[3]

    im_roi = im_gray[y_start: y_start + height, x_start: x_start + width]

    # Draw filled contours (removes overlapping items)
    mask = np.zeros((height, width)).astype(np.uint8)
    mask = cv2.drawContours(mask, [ctr], 0, (255, 255, 255), cv2.FILLED, offset=(-x_start, -y_start))
    roi = cv2.bitwise_and(mask.astype(np.uint8), im_roi.astype(np.uint8)).astype(np.uint8)

    hull = cv2.convexHull(ctr, returnPoints=False)
    defects = cv2.convexityDefects(ctr, hull)

    if hasattr(defects, 'shape'):
        mean_defect_dist = sum([d for s, e, f, d in defects[:, 0]]) / defects.shape[0]
        large_defects = [(s, e, f, d) for s, e, f, d in defects[:, 0] if d > mean_defect_dist]
    else:
        print("Roi is convex, no defects found")
        return [rect], [ctr]

    roi = np.pad(roi, ((y_start, 0), (x_start, 0)), 'constant', constant_values=(0, 0))
    for s, e, f, d in large_defects:
        dists = [(ctr[f][0][0] - ctr[x][0][0]) ** 2 + (ctr[f][0][1] - ctr[x][0][1]) ** 2 if x != f else 999 for
                 s, e, x, d in defects[:, 0]]
        closest_idx = np.argmin(dists)
        closest = defects[closest_idx, :]
        start = tuple(ctr[closest[0][2]][0])
        end = tuple(ctr[f][0])
        cv2.line(roi, start, end, [0, 0, 0], 2)

    # Re-Segment
    new_rects, new_ctrs = find_rects_ctrs(roi)
    return new_rects, new_ctrs


def sort_rects(rects_ctrs):
    rects_sort = sorted(rects_ctrs, key=lambda r: r[0][0])

    result = []
    for rect in rects_sort:
        # Filter out short artifacts
        if rect[0][3] < 10:
            continue

        # Filter to only rects with overlap
        prev = rect
        first = rect
        rects_sort_filt = []
        rects_sort_filt.append(prev)
        for x in rects_sort:
            if x == first:
                continue
            if calc_overlap(x[0], prev[0]) > (prev[0][3] / 3.0) and horiz_dist_ratio_check(prev[0], x[0]):
                rects_sort_filt.append(x)
                prev = x

        result += rects_sort_filt
        for rect_sorted in rects_sort_filt:
            rects_sort.remove(rect_sorted)

    return zip(*result)


def calc_overlap(rect_1, rect_2):
    rect_1_y = rect_1[1]
    rect_1_height = rect_1[3]

    rect_2_y = rect_2[1]
    rect_2_height = rect_2[3]

    overlap = min(rect_1_y + rect_1_height, rect_2_y + rect_2_height) - max(rect_1_y, rect_2_y)
    return overlap


def horiz_dist_ratio_check(r1, r2):
    return abs(r1[0] + r1[2] - r2[0]) < (3 * r1[2])


def draw_roi(rect, i, sorted_ctrs, im_gray, binary_roi):
    # Todo: take 2nd level of hierarchy, draw holes for 4, 6, 8, 9, 0 etc...
    x_start = rect[0]
    y_start = rect[1]
    width = rect[2]
    height = rect[3]

    # Try with either im_gray, or im_th_all_colors here
    im_roi = im_gray[y_start: y_start + height, x_start: x_start + width]

    # Draw filled contours (removes overlapping items)
    mask = np.zeros((height, width)).astype(np.uint8)
    mask = cv2.drawContours(mask, sorted_ctrs, i, (255, 255, 255), cv2.FILLED, offset=(-x_start, -y_start))
    if binary_roi:
        roi = mask
    else:
        roi = cv2.bitwise_and(mask.astype(np.uint8), im_roi.astype(np.uint8)).astype(np.uint8)

    # Pad ROI to square
    if height < width:
        padding = int((width - height) / 2.0)
        roi = np.pad(roi, ((padding, padding), (0, 0)), 'constant', constant_values=(0, 0))
    elif height > width:
        padding = int((height - width) / 2.0)
        roi = np.pad(roi, ((0, 0), (padding, padding)), 'constant', constant_values=(0, 0))

    roi_pad = np.pad(roi, (5, 5), 'constant', constant_values=(0, 0))
    roi_resized = cv2.resize(roi_pad, (28, 28), interpolation=cv2.INTER_NEAREST)
    return roi_resized


def deskew(img):
    SZ = 28
    affine_flags = cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR

    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11'] / m['mu02']
    M = np.float32([[1, skew, -0.5 * SZ * skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (SZ, SZ), flags=affine_flags)
    return img

