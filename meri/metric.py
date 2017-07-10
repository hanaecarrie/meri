""" Module that declare typical usefull metrics for image error measurement.
"""
import numpy as np
from sklearn.cluster import k_means
from scipy.ndimage.morphology import binary_closing
from skimage.measure import compare_ssim as _compare_ssim
from pisap.base.utils import min_max_normalize


def compute_ssim(test, ref):
    """ Return ssim(ref, test)

    Parameters:
    -----------
    ref: np.ndarray, the reference image

    test: np.ndarray, the tested image

    Return:
    -------
    ssim: float, the snr
    """
    #compute the ssim for all the image
    test = np.abs(test).astype('float64')
    ref = np.abs(ref).astype('float64')
    test = min_max_normalize(test)
    ref = min_max_normalize(ref)
    _, ssim = _compare_ssim(test, ref, full=True)
    # compute the mask for the foreground
    centroids, mask, _ = k_means(ref.flatten()[:, None], 2)
    if np.argmax(centroids) == 0:
        mask = np.abs(mask-1)
    mask = mask.reshape(*ref.shape)
    mask = binary_closing(mask, np.ones((5, 5)), iterations=3).astype('int')
    return (mask * ssim).sum() / mask.sum()


def compute_assim(test, ref):
    """ Return average ssim(ref, test)

    Parameters:
    -----------
    ref: np.ndarray, the reference image

    test: np.ndarray, the tested image

    Return:
    -------
    ssim: float, the snr
    """
    test = np.abs(test).astype('float64')
    ref = np.abs(ref).astype('float64')
    test = min_max_normalize(test)
    ref = min_max_normalize(ref)
    return _compare_ssim(test, ref)


def compute_snr(test, ref):
    """ Return 10 log_10( |test|_2^2 /  |test - ref|_2^2)

    Parameters:
    -----------
    ref: np.ndarray, the reference image

    test: np.ndarray, the tested image

    Return:
    -------
    snr: float, the snr
    """
    test = np.abs(test).astype('float64')
    ref = np.abs(ref).astype('float64')
    test = min_max_normalize(test)
    ref = min_max_normalize(ref)
    return 10.0 * np.log10(np.linalg.norm(test)**2 / np.linalg.norm(test-ref)**2)


def compute_psnr(test, ref):
    """ Return 10 log_10( max(abs(test))^2 /  |test - ref|_2^2)

    Parameters:
    -----------
    ref: np.ndarray, the reference image

    test: np.ndarray, the tested image

    Notes:
    ------
    Compute the metric only on magnetude.

    Return:
    -------
    psnr: float, the psnr
    """
    test = np.abs(test).astype('float64')
    ref = np.abs(ref).astype('float64')
    test = min_max_normalize(test)
    ref = min_max_normalize(ref)
    return 10.0 * np.log10(np.max(np.abs(test))**2 / np.linalg.norm(np.abs(test)-np.abs(ref))**2)


def compute_nrmse(test, ref, norm_type="euclidian"):
    """ Return |ref - test|_2 / norm(ref)

    Parameters:
    -----------
    ref: np.ndarray, the reference image

    test: np.ndarray, the tested image

    norm_type: str, ('euclidian', 'min_max'), default='euclidian', the type of normalization

    Return:
    -------
    nrmse: float, the nrmse
    """
    test = np.abs(test).astype('float64')
    ref = np.abs(ref).astype('float64')
    test = min_max_normalize(test)
    ref = min_max_normalize(ref)
    if norm_type == "euclidian":
        return np.linalg.norm(ref - test) / np.linalg.norm(ref)
    elif norm_type == "min_max":
        return np.linalg.norm(ref - test) / (np.max(ref) - np.min(ref))
    else:
        raise ValueError("'norm_type' not understood")
