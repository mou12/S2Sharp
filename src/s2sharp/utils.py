"""Shape conversion utilities: matrix <-> image format."""

import numpy as np


def conv2im(X: np.ndarray, nl: int, nc: int | None = None, L: int | None = None) -> np.ndarray:
    """Convert (L, n) matrix to (nl, nc, L) image.

    Replicates MATLAB conv2im: reshape(X', nl, nc, L).
    MATLAB's column-major reshape of the transposed matrix is equivalent
    to NumPy's default C-order reshape of X.T.
    """
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if L is None:
        L = X.shape[0]
    if nc is None:
        nc = X.shape[1] // nl
    # X is (L, n) where n = nl*nc
    # X.T is (n, L), reshape to (nl, nc, L) in C-order
    return X.T.reshape(nl, nc, L)


def conv2mat(X: np.ndarray) -> np.ndarray:
    """Convert (nl, nc, L) image to (L, n) matrix.

    Replicates MATLAB conv2mat: reshape(X, nl*nc, L)'.
    """
    if X.ndim == 2:
        # 2D image: treat as single band
        nl, nc = X.shape
        L = 1
        return X.reshape(nl * nc, L).T
    nl, nc, L = X.shape
    return X.reshape(nl * nc, L).T


def gaussian_kernel(size_x: int, size_y: int, sigma: float) -> np.ndarray:
    """Create a 2D Gaussian kernel, matching MATLAB fspecial('gaussian', [dx,dy], sigma).

    Parameters
    ----------
    size_x : int
        Width of kernel.
    size_y : int
        Height of kernel.
    sigma : float
        Standard deviation of Gaussian.

    Returns
    -------
    np.ndarray
        Normalized 2D Gaussian kernel of shape (size_y, size_x).
    """
    x = np.arange(size_x) - (size_x - 1) / 2
    y = np.arange(size_y) - (size_y - 1) / 2
    xx, yy = np.meshgrid(x, y)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return kernel / kernel.sum()
