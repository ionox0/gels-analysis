import os
import scipy
import warnings
from skimage import transform
from skimage.color import rgb2gray
from skimage.exposure import rescale_intensity
from skimage import data, img_as_float

warnings.filterwarnings("ignore")


ORIG_IMAGES = []

def load_april_2016_images():
    '''
    Read images data from data folder for April gels
    :return:
    '''
    filenames = os.listdir('../data/april_2016_gels_renamed/')
    filenames = [x for x in filenames if 'pep1' in x]
    filenames = [x for x in filenames if not 'big' in x]

    imgs_april = []
    for filename in filenames[0:20]:
        img = load_and_process_image('../data/april_2016_gels_renamed/' + filename)
        imgs_april.append(img)

    return imgs_april


def load_nov_2016_images():
    '''
    Read images data from data folder for November gels
    :return:
    '''
    imgs_nov = []
    imgs_nov_idx = [1, 6, 12, 21, 41, 42, 51, 52, 56, 83, 84, 89, 90, 96, 97, 106, 123, 131, 136, 152, 153, 156, 157]  # 7, 22

    nov_imgs = [f for f in os.listdir('../data/gels_nov_2016') if not 'tore' in f]
    nov_filenames = ['../data/gels_nov_2016/{}'.format(f) for f in nov_imgs]

    for idx in imgs_nov_idx[0:2]:
        filename = nov_filenames[idx]
        img = load_and_process_image(filename)
        imgs_nov.append(img)

    return imgs_nov


def load_and_process_image(filename):
    '''
    Read an image and resize to consistent sizing (hard coded for now)
    :param filename:
    :return:
    '''
    shape = (1276, 2100)

    ORIG_IMAGES.append(scipy.misc.imread(filename))

    cur_im = data.imread(filename, flatten=True)
    cur_im = img_as_float(cur_im)
    cur_im = rescale_intensity(cur_im)
    cur_im = rgb2gray(cur_im)

    cur_im = transform.resize(cur_im, output_shape=shape)  # todo
    return cur_im