from __future__ import print_function
from builtins import zip
from builtins import range
from past.builtins import xrange

import matplotlib
import numpy as np
from scipy.ndimage import uniform_filter


def extract_features(imgs, feature_fns, verbose=False):
    """
    Given pixel data for images and several feature functions that can operate on
    single images, apply all feature functions to all images, concatenating the
    feature vectors for each image and storing the features for all images in
    a single matrix.

    Inputs:
    - imgs (X_train): N x H X W X C array of pixel data for N images .
    - feature_fns: List of k feature functions. The ith feature function should
      take as input an H x W x D array and return a (one-dimensional) array of
      length F_i.
    - verbose: Boolean; if true, print progress.

    Returns:
    An array of shape (N, F_1 + ... + F_k) where each column is the concatenation
    of all features for a single image.
    """
    num_images = imgs.shape[0]
    if num_images == 0:
        return np.array([])

    # Use the first image to determine feature dimensions
    feature_dims = []
    first_image_features = []
    
    # iterate over List of k feature functions:[hog_feature, color_histogram_hsv]
    for feature_fn in feature_fns:
        # imgs[0].squeeze() - take first image remove first dimension (1,32,32,3) -> (32,32,3)  
        feats = feature_fn(imgs[0].squeeze())
        assert len(feats.shape) == 1, "Feature functions must be one-dimensional"
        feature_dims.append(feats.size)
        first_image_features.append(feats)

    # Now that we know the dimensions of the features, we can allocate a single
    # big array to store all features as columns.
    total_feature_dim = sum(feature_dims)
    imgs_features = np.zeros((num_images, total_feature_dim))
    imgs_features[0] = np.hstack(first_image_features).T

    # Extract features for the rest of the images.
    for i in range(1, num_images):
        idx = 0
        for feature_fn, feature_dim in zip(feature_fns, feature_dims):
            next_idx = idx + feature_dim
            imgs_features[i, idx:next_idx] = feature_fn(imgs[i].squeeze())
            idx = next_idx
        if verbose and i % 1000 == 999:
            print("Done extracting features for %d / %d images" % (i + 1, num_images))

    return imgs_features


def rgb2gray(rgb):
    """Convert RGB image to grayscale

      Parameters:
        rgb : RGB image

      Returns:
        gray : grayscale image

    """
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.144])


def hog_feature(im):
    """Compute Histogram of Gradient (HOG) feature for an image

         Modified from skimage.feature.hog
         http://pydoc.net/Python/scikits-image/0.4.2/skimage.feature.hog

       Reference:
         Histograms of Oriented Gradients for Human Detection
         Navneet Dalal and Bill Triggs, CVPR 2005

      Parameters:
        im : an input grayscale or rgb image

      Returns:
        feat: Histogram of Gradient (HOG) feature

    """

    # convert rgb to grayscale if needed
    if im.ndim == 3:
      # remove the rgb dim
        image = rgb2gray(im)
    else:
      #  Convert to array with at least two dimension; 1-dim inputs converted to 2-dim arrays
      # higher-dimensional inputs are preserved.
        image = np.at_least_2d(im)
    
    sx, sy = image.shape  # image size (32,32)
    orientations = 9  # number of gradient bins
    cx, cy = (8, 8)  # pixels per cell (32/8)*(32/8) = 4*4 = 16 cells total
    n_cellsx = int(np.floor(sx / cx))  # number of cells in x
    n_cellsy = int(np.floor(sy / cy))  # number of cells in y
    
    # compute image gradiant on x,y directions and comput total grad direction and intensity 
    gx = np.zeros(image.shape) 
    gy = np.zeros(image.shape)
    # Calculate the n=1 discrete difference along the given axis:
    gx[:, :-1] = np.diff(image, n=1, axis=1)  # compute gradient on x-direction
    gy[:-1, :] = np.diff(image, n=1, axis=0)  # compute gradient on y-direction
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)  # gradient magnitude
    # Element-wise arc-tan of gy/gx transformed to degrees and rotated by 90 deg (relative to positive x axis):
    grad_ori = np.arctan2(gy, (gx + 1e-15)) * (180 / np.pi) + 90  # gradient orientation

   
    # compute orientations integral images
    orientation_histogram = np.zeros((n_cellsx, n_cellsy, orientations))
    for i in range(orientations):
        # create new integral image for this orientation
        # isolate orientations in this range
        # grad_ori <18,36..180 and grad_ori >0,18..162 i.e 0-18..162-180
        
        temp_ori = np.where(grad_ori < 180 / orientations * (i + 1), grad_ori, 0)
        temp_ori = np.where(grad_ori >= 180 / orientations * i, temp_ori, 0)
        # select magnitudes for those orientations
        cond2 = temp_ori > 0
        temp_mag = np.where(cond2, grad_mag, 0) # same size as image with grad magnitude where grad oriantation is between the limits 
       
        # uniform_filter is a multidimesional sequence of 1-D uniform filters; 
        # returning the same dim as input (32,32)
        # take the moddle of each cell i.e from cx/2 jump by cx and same in y recieve (4,4)
        # transpose back 
        orientation_histogram[:, :, i] = uniform_filter(temp_mag, size=(cx, cy))[
            round(cx / 2) :: cx, round(cy / 2) :: cy
        ].T
        
        # get (4,4) cells for each orientations (4,4,9) and flaten to 4*4*9=144 vector
        # ("Feature functions must be one-dimensional")
    return orientation_histogram.ravel()


def color_histogram_hsv(im, nbin=10, xmin=0, xmax=255, normalized=True):
    """
    Compute color histogram for an image using hue.

    Inputs:
    - im: H x W x C array of pixel data for an RGB image.
    - nbin: Number of histogram bins. (default: 10)
    - xmin: Minimum pixel value (default: 0)
    - xmax: Maximum pixel value (default: 255)
    - normalized: Whether to normalize the histogram (default: True)

    Returns:
      1D vector of length nbin giving the color histogram over the hue of the
      input image.
    """
    ndim = im.ndim
    bins = np.linspace(xmin, xmax, nbin + 1)
    hsv = matplotlib.colors.rgb_to_hsv(im / xmax) * xmax #from rgb to hsv with same dimensions (32,32,3)
    # hsv[:, :, 0] is the image hue (color angle)
    imhist, bin_edges = np.histogram(hsv[:, :, 0], bins=bins, density=normalized)
    # nbin=10; imhist 10 vector hue hist normalized by bin width (normalized color angle hist)
    imhist = imhist * np.diff(bin_edges)

    
    return imhist


# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

pass

# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

import inspect, re

def varname(p):
  for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
    m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
    if m:
      return m.group(1)



def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000):
    # Load the raw CIFAR-10 data
    cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'

    # Cleaning up variables to prevent loading data multiple times (which may cause memory issue)
    try:
       del X_train, y_train
       del X_test, y_test
       print('Clear previously loaded data.')
    except:
       pass

    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
    
    # Subsample the data
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]

    print(varname(X_val),X_val.shape," ",varname(y_val),y_val.shape)
    print(varname(X_train),X_train.shape," ",varname(y_train),y_train.shape)
    print(varname(X_test),X_test.shape," ",varname(y_test),y_test.shape)
    return X_train, y_train, X_val, y_val, X_test, y_test



if __name__ == "__main__":
  from data_utils import load_CIFAR10
  X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()

  num_color_bins = 10 # Number of bins in the color histogram
  feature_fns = [hog_feature, lambda img: color_histogram_hsv(img, nbin=num_color_bins)]
  X_train_feats = extract_features(X_train, feature_fns, verbose=True)
  X_val_feats = extract_features(X_val, feature_fns)
  X_test_feats = extract_features(X_test, feature_fns)

  # Preprocessing: Subtract the mean feature
  mean_feat = np.mean(X_train_feats, axis=0, keepdims=True)
  X_train_feats -= mean_feat
  X_val_feats -= mean_feat
  X_test_feats -= mean_feat

  # Preprocessing: Divide by standard deviation. This ensures that each feature
  # has roughly the same scale.
  std_feat = np.std(X_train_feats, axis=0, keepdims=True)
  X_train_feats /= std_feat
  X_val_feats /= std_feat
  X_test_feats /= std_feat

  # Preprocessing: Add a bias dimension
  X_train_feats = np.hstack([X_train_feats, np.ones((X_train_feats.shape[0], 1))])
  X_val_feats = np.hstack([X_val_feats, np.ones((X_val_feats.shape[0], 1))])
  X_test_feats = np.hstack([X_test_feats, np.ones((X_test_feats.shape[0], 1))])