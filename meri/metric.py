""" Module that declare typical usefull metrics for image error measurement.
"""
import numpy as np
from sklearn.cluster import k_means
from scipy.ndimage.morphology import binary_closing
from skimage.measure import compare_ssim as _compare_ssim
import matplotlib.pyplot as plt


def _min_max_normalize(img):
    """ Center and normalize the given array.

    Parameters:
    ----------
    img: np.ndarray

    Return:
    ------
    im: np.ndarray
        the min-max normalized image.
    """
    min_img = img.min()
    max_img = img.max()
    return (img - min_img) / (max_img - min_img)


def compute_ssim(test, ref, mask="auto", disp=False):
    """ Return SSIM

    Parameters:
    -----------
    ref: np.ndarray, the reference image

    test: np.ndarray, the tested image

    mask: np.ndarray, the mask for the ROI

    disp: bool (default False), if True display the mask.

    Notes:
    ------
    Compute the metric only on magnetude.

    Return:
    -------
    ssim: float, the snr
    """
    test = np.abs(test).astype('float64')
    ref = np.abs(ref).astype('float64')
    test = _min_max_normalize(test)
    ref = _min_max_normalize(ref)
    assim, ssim = _compare_ssim(test, ref, full=True)
    if (not isinstance(mask, np.ndarray)) and (mask not in ["auto", None]):
        raise ValueError("mask should be None, 'auto' or a np.ndarray,"
                         " got '{0}' instead.".format(mask))
    if mask is None:
        return assim
    if mask == "auto":
        centroids, mask, _ = k_means(ref.flatten()[:, None], 2)
        if np.argmax(centroids) == 0:
            mask = np.abs(mask-1)
        mask = mask.reshape(*ref.shape)
        mask = binary_closing(mask, np.ones((5, 5)), iterations=4).astype('int')
    if disp:
        plt.matshow(0.5 * (mask + ref), cmap='gray')
        plt.show()
    return (mask * ssim).sum() / mask.sum()


def compute_snr(test, ref):
    """ Return SNR

    Parameters:
    -----------
    ref: np.ndarray, the reference image

    test: np.ndarray, the tested image

    Notes:
    ------
    Compute the metric only on magnetude.

    Return:
    -------
    snr: float, the snr
    """
    test = np.abs(test).astype('float64')
    ref = np.abs(ref).astype('float64')
    test = _min_max_normalize(test)
    ref = _min_max_normalize(ref)
    num = np.mean(np.square(ref))
    deno = compute_mse(test, ref)
    return 10.0 * np.log10(num / deno)


def compute_psnr(test, ref):
    """ Return PSNR

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
    test = _min_max_normalize(test)
    ref = _min_max_normalize(ref)
    num = np.max(np.abs(ref))
    deno = compute_mse(test, ref)
    return 10.0 * np.log10(num / deno)


def compute_mse(test, ref):
    """ Return 1/N * |ref - test|_2

    Parameters:
    -----------
    ref: np.ndarray, the reference image

    test: np.ndarray, the tested image

    Notes:
    -----
    Compute the metric only on magnetude.

    Return:
    -------
    mse: float, the mse
    """
    test = np.abs(test).astype('float64')
    ref = np.abs(ref).astype('float64')
    test = _min_max_normalize(test)
    ref = _min_max_normalize(ref)
    return np.mean(np.square(test - ref))


def compute_nrmse(test, ref):
    """ Return NRMSE

    Parameters:
    -----------
    ref: np.ndarray, the reference image

    test: np.ndarray, the tested image

    Notes:
    -----
    Compute the metric only on magnetude.

    Return:
    -------
    nrmse: float, the nrmse
    """
    test = np.abs(test).astype('float64')
    ref = np.abs(ref).astype('float64')
    test = _min_max_normalize(test)
    ref = _min_max_normalize(ref)
    num = np.sqrt(compute_mse(test, ref))
    deno = np.sqrt(np.mean((np.square(test))))
    return num / deno
