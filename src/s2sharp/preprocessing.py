"""Data preparation: normalization, subsampling, and weight computation."""

import numpy as np

from .utils import conv2im, conv2mat


def normalize_data(
    bands: list[np.ndarray],
) -> tuple[list[np.ndarray], np.ndarray]:
    """Normalize each band to unit mean-squared power.

    Replicates MATLAB normaliseData for cell array input.

    Parameters
    ----------
    bands : list[np.ndarray]
        List of L band images (possibly different sizes).

    Returns
    -------
    tuple[list[np.ndarray], np.ndarray]
        Normalized bands and normalization factors (av), shape (L,).
    """
    nb = len(bands)
    av = np.zeros(nb)
    normalized = []
    for i in range(nb):
        av[i] = np.mean(bands[i].astype(np.float64) ** 2)
        normalized.append(np.sqrt(bands[i].astype(np.float64) ** 2 / av[i]))
    return normalized, av


def normalize_data_image(
    image: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize a 3D image array (nl, nc, L) to unit mean-squared power per band.

    Replicates MATLAB normaliseData for 3D array input.

    Parameters
    ----------
    image : np.ndarray
        Image of shape (nl, nc, L).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Normalized image and normalization factors, shape (L,).
    """
    nb = image.shape[2]
    av = np.zeros(nb)
    result = image.astype(np.float64).copy()
    for i in range(nb):
        av[i] = np.mean(result[:, :, i] ** 2)
        result[:, :, i] = np.sqrt(result[:, :, i] ** 2 / av[i])
    return result, av


def unnormalize_data(image: np.ndarray, av: np.ndarray) -> np.ndarray:
    """Reverse normalization on a 3D image array.

    Replicates MATLAB unnormaliseData for 3D array input.

    Parameters
    ----------
    image : np.ndarray
        Normalized image of shape (nl, nc, L).
    av : np.ndarray
        Normalization factors, shape (L,).

    Returns
    -------
    np.ndarray
        Unnormalized image.
    """
    result = image.copy()
    nb = result.shape[2]
    for i in range(nb):
        result[:, :, i] = np.sqrt(result[:, :, i] ** 2 * av[i])
    return result


def create_subsampling(
    bands: list[np.ndarray],
    d: np.ndarray,
    nl: int,
    nc: int,
    L: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build subsampling masks and observed data matrix.

    Replicates MATLAB createSubsampling (S2sharp.m lines 422-434).

    Parameters
    ----------
    bands : list[np.ndarray]
        Normalized band images.
    d : np.ndarray
        Downsampling factors, shape (L,).
    nl, nc : int
        Image dimensions at 10m resolution.
    L : int
        Number of bands.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        M : subsampling masks, shape (nl, nc, L)
        Y : observed data matrix, shape (L, nl*nc)
    """
    M = np.zeros((nl, nc, L))
    n = nl * nc
    Y = np.zeros((L, n))

    for i in range(L):
        di = d[i]
        # Create subsampling pattern using Kronecker product
        im = np.ones((nl // di, nc // di))
        maux = np.zeros((di, di))
        maux[0, 0] = 1
        M[:, :, i] = np.kron(im, maux)

        # Find indices where mask is 1
        indexes = np.where(M[:, :, i].ravel() == 1)[0]

        # Fill Y with observed data at those indices
        band_mat = conv2mat(bands[i])  # (1, nl_band * nc_band)
        Y[i, indexes] = band_mat.ravel()

    return M, Y


def compute_weights(
    Y: np.ndarray,
    d: np.ndarray,
    sigmas: float,
    nl: int,
) -> np.ndarray:
    """Compute adaptive regularization weights from high-resolution band gradients.

    Replicates MATLAB computeWeights (S2sharp.m lines 447-460).

    Parameters
    ----------
    Y : np.ndarray
        Observed data matrix, shape (L, n).
    d : np.ndarray
        Downsampling factors, shape (L,).
    sigmas : float
        Weight decay parameter.
    nl : int
        Number of image rows.

    Returns
    -------
    np.ndarray
        Weight matrix, shape (1, n).
    """
    # Find high-resolution bands (d == 1)
    hr_bands = np.where(d == 1)[0]

    nc = Y.shape[1] // nl
    grad_all = np.zeros((nl, nc, len(hr_bands)))

    # MATLAB's imgradient with 'intermediate' method uses 1D forward differences
    # with replicate boundary (equivalent to zero gradient at last row/col).
    for idx, i in enumerate(hr_bands):
        img = conv2im(Y[i:i + 1, :], nl, nc, 1).squeeze()
        gx = np.zeros_like(img)
        gy = np.zeros_like(img)
        gx[:, :-1] = img[:, 1:] - img[:, :-1]
        gy[:-1, :] = img[1:, :] - img[:-1, :]
        grad_all[:, :, idx] = np.hypot(gx, gy) ** 2

    # Max gradient across HR bands, then sqrt
    grad = np.sqrt(np.max(grad_all, axis=2))

    # Normalize by 95th percentile (method='hazen' matches MATLAB's quantile)
    q95 = np.quantile(grad.ravel(), 0.95, method='hazen')
    if q95 > 0:
        grad = grad / q95

    # Compute weights
    Wim = np.exp(-grad ** 2 / (2 * sigmas ** 2))
    Wim[Wim < 0.5] = 0.5

    W = conv2mat(Wim)  # (1, n)
    return W
