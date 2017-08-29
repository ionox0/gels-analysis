import matplotlib

from skimage import data


def extract_roi(image, roi_metadata):
    roi = image[
          roi_metadata['y_start']: roi_metadata['y_end'],
          roi_metadata['x_start']: roi_metadata['x_end']
          ]
    return roi


def create_train_images(filename, rois, labels):
    path = './full_gels/' + filename
    img = data.imread(path, False)

    for i, roi_metadata in enumerate(rois):
        lane = extract_roi(img, roi_metadata)

        label = labels[i]
        res_filename = './train_images/' + str(i) + '_' + label + '_' + filename
        matplotlib.image.imsave(res_filename, lane)
