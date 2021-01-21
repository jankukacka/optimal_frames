# ---------------------------------------------------------------------
#  File: motion_metrics.py
#  Author: Jan Kukacka
#  Date: 1/2021
# ---------------------------------------------------------------------
#  Motion metrics
# ---------------------------------------------------------------------

import numpy as np
from scipy.stats import entropy
from skimage.metrics import structural_similarity as ssim
from functools import lru_cache


def cross_correlation(x,y):
    '''
    Standard cross-correlation
    '''
    return -np.sum(x*y)


def norm_cross_correlation(x,y):
    '''
    Normalized cross-correlation
    https://en.wikipedia.org/wiki/Cross-correlation
    '''
    a = x/x.std()
    b = y/y.std()
    return -np.sum(a*b)


def zeronorm_cross_correlation(x,y):
    '''
    Zero-normalized cross-correlation
    https://en.wikipedia.org/wiki/Cross-correlation#Zero-normalized_cross-correlation_(ZNCC)
    '''
    a = (x-x.mean())/x.std()
    b = (y-y.mean())/y.std()
    return -np.sum(a*b)


def root_mean_squared_error(x,y):
    '''
    Root of mean squared error
    '''
    return np.sqrt(np.sum((x-y)**2))


def wasserstein(x,y, dist_map=None):
    '''
    Wasserstein distance
    '''
    try:
        import ot
    except Exception as e:
        raise Exception('To use wasserstein distance, install the "ot" package.')

    if dist_map is None:
        dist_map = _dm(x.shape)

    reg = .1
    max_iter = 10
    im1_where = np.where(x)
    im2_where = np.where(y)
    im1 = np.ravel_multi_index(im1_where, dims=x.shape)
    im2 = np.ravel_multi_index(im2_where, dims=y.shape)
    return ot.sinkhorn2(x[im1_where]/np.sum(x), y[im2_where]/np.sum(y), dist_map[im1][:,im2], reg, numItermax=max_iter)[0]


def structural_similarity(x,y,**kwargs):
    '''
    Structural similarity
    '''
    return -ssim(x,y,**kwargs)


def norm_mutual_information(x, y, bins=100):
    '''
    Normalized mutual information metric
    from https://github.com/scikit-image/scikit-image/blob/e1c7ed338433349fee77e6e5d36f2e30690ba812/skimage/metrics/simple_metrics.py
    '''
    hist, bin_edges = np.histogramdd([np.ravel(x),
                                      np.ravel(y)],
                                     bins=bins, density=True)

    H_im_true = entropy(np.sum(hist, axis=0))
    H_im_test = entropy(np.sum(hist, axis=1))
    H_true_test = entropy(np.ravel(hist))

    return -(H_im_true + H_im_test) / H_true_test


## LRU cache to evaluate this only once and lazy
@lru_cache(1)
def _dm(shape=(50,50)):
    # print('Computing default distance matrix')
    xx = np.arange(shape[1])[None]
    yy = np.arange(shape[0])[:,None]
    xy = np.stack(np.broadcast_arrays(xx,yy), axis=-1).reshape(-1,2)
    diff = xy[None] - xy[:,None]
    dist_map = np.sqrt(np.sum(diff*diff, axis=-1))
    return dist_map


_all_metrics = {
    'xc': cross_correlation,
    'nxc': norm_cross_correlation,
    'znxc': zeronorm_cross_correlation,
    'rmse': root_mean_squared_error,
    'wass': wasserstein,
    'ssim': structural_similarity,
    'nmi': norm_mutual_information
}
