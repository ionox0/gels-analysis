
# coding: utf-8

# In[4]:

import PyPDF2
from PIL import Image

import sys
import warnings
import matplotlib
import numpy as np
from os import path
from matplotlib import pyplot as plt
warnings.filterwarnings("ignore")


# ### New PDF Extraction

# In[ ]:

def extract_images_from_pdf(filename, num_pages, dest_dir):
    number = 0

    def recurse(page, xObject):
        global number

        xObject = xObject['/Resources']['/XObject'].getObject()

        for obj in xObject:

            if xObject[obj]['/Subtype'] == '/Image':
                size = (xObject[obj]['/Width'], xObject[obj]['/Height'])
                data = xObject[obj].getData()

                if xObject[obj]['/ColorSpace'] == '/DeviceRGB':
                    mode = "RGB"
                else:
                    # todo - currently manually set to RGB
                    mode = "RGB"

                imagename = "%s - p. %s"%(obj[1:], p)

                if xObject[obj]['/Filter'] == '/FlateDecode':
                    img = Image.frombytes(mode, size, data)
                    img.save(dest_dir + imagename + ".png")
                    number += 1
                    
                # todo
#                 elif xObject[obj]['/Filter'] == '/DCTDecode':
#                     img = open(imagename + ".jpg", "wb")
#                     img.write(data)
#                     img.close()
#                     number += 1
#                 elif xObject[obj]['/Filter'] == '/JPXDecode':
#                     img = open(imagename + ".jp2", "wb")
#                     img.write(data)
#                     img.close()
#                     number += 1
            else:
                recurse(page, xObject[obj])

    abspath = path.abspath(filename)
    pdf_file = PyPDF2.PdfFileReader(open(filename, "rb"))

    for p in range(num_pages):    
        page0 = pdf_file.getPage(p-1)
        recurse(p, page0)

    print('%s extracted images'% number)
    
    
extract_images_from_pdf('../data/GelsNov2016.pdf', 162, '../data/gels_nov_2016/')


# ### Collect Type I gels

# In[5]:

from skimage import data
from skimage import transform
from skimage.util import img_as_float
from skimage.color import rgb2gray
from skimage.exposure import rescale_intensity


imgs_blue = []
imgs_blue_idx = [1,6,7,12,21,22,41,42,51,52,56,83,84,89,90,96,97,106,123,131,136,152,153,156,157]

shape = (1276, 2100)

for idx in imgs_blue_idx:
    cur_im = data.imread('../data/gels_nov_2016/Im{} - p. {}.png'.format(idx, idx), flatten=True)
    
    cur_im = img_as_float(cur_im)
    cur_im = rescale_intensity(cur_im)
    cur_im = rgb2gray(cur_im)
    
    cur_im = transform.resize(cur_im, output_shape=shape) # todo
    
    imgs_blue.append(cur_im)
    
len(imgs_blue)


# ### Image Alignment - ORB + RANSAC

# In[6]:

from skimage.feature import ORB, match_descriptors
from skimage.transform import ProjectiveTransform, AffineTransform, EuclideanTransform # !!!!!!!!!!
from skimage.measure import ransac

from skimage.color import gray2rgb
from skimage.exposure import rescale_intensity
from skimage.transform import warp
from skimage.transform import SimilarityTransform


img_1 = imgs_blue[0]


# In[44]:

def add_alpha(image, background=-1):
    """Add an alpha layer to the image.

    The alpha layer is set to 1 for foreground
    and 0 for background.
    """
    rgb = gray2rgb(image)
    alpha = (image != background)
    return np.dstack((rgb, alpha))


def align_images(img_1, img_2):
    orb = ORB(n_keypoints=500, fast_threshold=0.05)

    orb.detect_and_extract(img_1)
    keypoints1 = orb.keypoints
    descriptors1 = orb.descriptors

    orb.detect_and_extract(img_2)
    keypoints2 = orb.keypoints
    descriptors2 = orb.descriptors

    matches12 = match_descriptors(descriptors1,
                                  descriptors2,
                                  cross_check=True)

    # Select keypoints from the source (image to be
    # registered) and target (reference image).
    src = keypoints2[matches12[:, 1]][:, ::-1]
    dst = keypoints1[matches12[:, 0]][:, ::-1]

    model_robust, inliers = ransac((src, dst), EuclideanTransform,
               min_samples=4, residual_threshold=2)

    r, c = img_1.shape[:2]

    # Note that transformations take coordinates in
    # (x, y) format, not (row, column), in order to be
    # consistent with most literature.
    corners = np.array([[0, 0],
                        [0, r],
                        [c, 0],
                        [c, r]])

    # Warp the image corners to their new positions.
    warped_corners = model_robust(corners)

    # Find the extents of both the reference image and
    # the warped target image.
    all_corners = np.vstack((warped_corners, corners))

    corner_min = np.min(all_corners, axis=0)
    corner_max = np.max(all_corners, axis=0)

    output_shape = (corner_max - corner_min)
    output_shape += np.abs(corner_min)
    output_shape = output_shape[::-1].astype(int) # todo - check
    
    if np.min(output_shape) < 0: # todo - means no alignment found?
        return None

    offset = SimilarityTransform(translation=-corner_min)
    
    image0_ = warp(img_1, offset.inverse,
                   output_shape=output_shape, cval=-1)

    image1_ = warp(img_2, (offset + model_robust).inverse,
                   output_shape=output_shape, cval=-1)

    image0_alpha = add_alpha(image0_)
    image1_alpha = add_alpha(image1_)

    merged = (image0_alpha + image1_alpha)
    alpha = merged[..., 3]

    # The summed alpha layers give us an indication of
    # how many images were combined to make up each
    # pixel. Divide by the number of images to get
    # an average.
    merged /= np.maximum(alpha, 1)[..., np.newaxis]
    
    return merged


# In[49]:

merges_with_one = [align_images(img_1, x) for x in imgs_blue]

for i, m in enumerate(merges_with_one):
    if m != None:
        orig_idx = imgs_blue_idx[i]
        result_filename = 'merged_{}.jpg'.format(orig_idx)
        matplotlib.image.imsave('./alignments/' + result_filename, m)


# In[ ]:



