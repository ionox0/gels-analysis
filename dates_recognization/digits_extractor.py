import copy
# import keras
from skimage.morphology import opening, closing

from preprocessor import *



# clf, pp = joblib.load('./models/digits_cls.pkl')
model = keras.models.load_model('./models/double_digits_cnn_nonbinary')
model_single = keras.models.load_model('./models/digits_cnn_new')


def extract_numbers(
        im,
        grayscale=False,
        thresh=30,
        invert=True,
        blue_thresh=False,
        binary_roi=False,
        separate_c=False,
        blur=0,
        contrast_inc=0,
        brightness_inc=0,
        opening_shape=None,
        closing_shape=None,
        dilation_size=0,
        erosion_size=0,
        should_deskew=False):

    im_c = im.copy()

    # Convert to grayscale
    if grayscale:
        im_gray = cv2.cvtColor(im_c, cv2.COLOR_BGR2GRAY)
    else:
        # Just use blue channel
        im_gray = im_c.astype(np.uint8)[:, :, 2]

    # Invert the image (todo - use im_gray?)
    if invert:
        im_gray = (255 - im_gray)

    # Blue thresholding uses original image,
    # regular thresholding uses im_gray
    if blue_thresh:
        im_th = thresh_img(im, thresh, blue_thresh=True)
    else:
        im_th = thresh_img(im_gray, thresh, blue_thresh=False)

    # Opening / Closing (also for contours)
    if hasattr(opening_shape, 'shape'):
        im_th = opening(im_th, opening_shape)
    if hasattr(closing_shape, 'shape'):
        im_th = closing(im_th, closing_shape)

    # Apply all preprocessing
    im_gray = im_preprocessing(im_gray, blur, brightness_inc, contrast_inc, dilation_size, erosion_size)

    rects, ctrs = find_rects_ctrs(im_th)
    # Skip short artifacts
    ctrs = [c for i, c in enumerate(ctrs) if rects[i][3] > 10]
    rects = [r for i, r in enumerate(rects) if r[3] > 10]
    # Skip skinny artifacts
    ctrs = [c for i, c in enumerate(ctrs) if rects[i][2] > 2]
    rects = [r for i, r in enumerate(rects) if r[2] > 2]

    # Separate connected digits
    if separate_c:
        rects, ctrs = split_dbls(rects, ctrs, im_gray)

    # Sorted order bounding boxes
    sorted_rects, sorted_ctrs = sort_rects(zip(rects, ctrs))

    rois = []
    probs = []
    date_possibs = []
    cur_date_possibs = []
    prev_not_one_width = 99999
    prev_end_x = sorted_rects[0][0] + sorted_rects[0][2]
    prev_end_y = sorted_rects[0][1] + sorted_rects[0][3]

    # For each rectangular region,
    # extract the roi from the preprocessed image,
    # and predict the digit using classifier
    for i, rect in enumerate(sorted_rects):
        x_start = rect[0]
        y_start = rect[1]
        width = rect[2]
        height = rect[3]
        # Skip short artifacts
        if rect[3] < 10: continue
        # Skip long artifacts
        if rect[2] > 100: continue

        roi_pad = draw_roi(rect, i, sorted_ctrs, im_gray, binary_roi)

        # Create new date possib (for newlines)
        newline = True
        if abs(x_start - prev_end_x) < 80 and abs(y_start - prev_end_y) < 80:
            newline = False
        if newline:
            date_possibs += cur_date_possibs
            cur_date_possibs = []

        dbl = False
        # Differentiate single from connected digits
        if width > 1.2 * prev_not_one_width:
            dbl = True
            roi_resized = cv2.resize(roi_pad, (38, 28), interpolation=cv2.INTER_NEAREST)
            roi_cnn = np.expand_dims(roi_resized, axis=2)

            prob = model.predict_proba(np.array([roi_cnn]), verbose=0)
            dbl_nbr = np.argmax(prob)

            nbr_prob = prob[0]

            # copy the current possibs
            new_cur_date_possibs = copy.deepcopy(cur_date_possibs)

            if len(new_cur_date_possibs) == 0:
                new_cur_date_possibs = [[dbl_nbr]]
            else:
                for date_possib in new_cur_date_possibs:
                    date_possib.append(dbl_nbr)

        # Deskew
        if should_deskew:
            roi_pad = deskew(roi_pad)

        roi_cnn = np.expand_dims(roi_pad, axis=2)
        prob = model_single.predict_proba(np.array([roi_cnn]), verbose=0)
        nbr = np.argmax(prob)
        nbr_prob = prob[0]

        if len(cur_date_possibs) == 0:
            cur_date_possibs = [[nbr]]
        else:
            for date_possib in cur_date_possibs:
                date_possib.append(nbr)

        if dbl:
            cur_date_possibs += new_cur_date_possibs

        prev_end_x = x_start
        prev_end_y = y_start

        # Mark the roi, label, and hierarchy
        cv2.rectangle(im_c, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 100, 255), 1)
        cv2.putText(im_c, str(int(nbr)), (rect[0], rect[1]), cv2.FONT_ITALIC, 0.4, (255, 0, 100), 1)
        # cv2.putText(im_c, str(hierarchy[0][i]) + str(int(nbr)), (rect[0], rect[1] - (250 - i*20)), cv2.FONT_ITALIC, 0.4, (randint(0,255), 0, 255), 1)

        if dbl:
            dbl_label = str(int(dbl_nbr))
            cv2.putText(im_c, dbl_label, (rect[0], rect[1] - 15), cv2.FONT_ITALIC, 0.3, (255, 0, 200), 1)

        probs.append(0)
        rois.append(roi_pad)

        if nbr != 1:
            prev_not_one_width = width

    # Append the final date possibility
    date_possibs += cur_date_possibs

    return im_c, rois, date_possibs, probs
