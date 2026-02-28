"""Shape conversion utilities: matrix <-> image format."""

import math

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


def matlab_round(x):
    """Round half-up, matching MATLAB's round() behavior.

    Python's round() uses banker's rounding (round half to even),
    e.g. round(204.5) = 204, but MATLAB gives 205.
    """
    return int(math.floor(x + 0.5))


def _keys_cubic(x, a=-0.5):
    """Keys cubic interpolation kernel (Catmull-Rom when a=-0.5).

    This is the kernel used by MATLAB's imresize for cubic interpolation.
    """
    absx = np.abs(x)
    absx2 = absx * absx
    absx3 = absx2 * absx

    result = np.zeros_like(x, dtype=np.float64)

    mask1 = absx <= 1
    result[mask1] = ((a + 2) * absx3[mask1]
                     - (a + 3) * absx2[mask1] + 1)

    mask2 = (absx > 1) & (absx <= 2)
    result[mask2] = (a * absx3[mask2]
                     - 5 * a * absx2[mask2]
                     + 8 * a * absx[mask2]
                     - 4 * a)

    return result


def matlab_imresize(img, scale):
    """Resize a 2D image matching MATLAB's imresize(img, scale) with bicubic interpolation.

    Uses the Keys cubic kernel (a=-0.5, Catmull-Rom) with separable 1D interpolation.
    For magnification (scale > 1), no anti-aliasing filter is applied (matching MATLAB).

    Parameters
    ----------
    img : np.ndarray
        Input 2D image.
    scale : int or float
        Scale factor. Must be >= 1 (magnification only).

    Returns
    -------
    np.ndarray
        Resized image.
    """
    in_h, in_w = img.shape
    out_h = int(matlab_round(in_h * scale))
    out_w = int(matlab_round(in_w * scale))

    # Resize along columns (height) first, then rows (width)
    temp = _resize_1d(img, in_h, out_h, axis=0)
    result = _resize_1d(temp, in_w, out_w, axis=1)
    return result


def _resize_1d(img, in_size, out_size, axis):
    """Perform 1D interpolation along the specified axis using Keys cubic kernel.

    Matches MATLAB's imresize output grid and kernel exactly.
    """
    # MATLAB's output coordinate mapping for magnification (scale >= 1):
    # out_coords in [1, out_size] mapped to input coords via:
    #   u = (out_coord - 0.5) / scale + 0.5
    # where scale = out_size / in_size
    scale = out_size / in_size
    kernel_width = 4.0  # Keys cubic has support [-2, 2]

    # Output coordinates (1-based, matching MATLAB)
    out_coords = np.arange(1, out_size + 1, dtype=np.float64)
    # Map to input coordinates (1-based)
    u = (out_coords - 0.5) / scale + 0.5

    # For each output pixel, find contributing input pixels
    # Kernel support is [-2, 2] for Keys cubic without anti-aliasing
    left = np.floor(u - kernel_width / 2).astype(int)  # leftmost contributing pixel

    # Build indices and weights
    n_contrib = int(np.ceil(kernel_width)) + 2  # 6, matching MATLAB
    indices = np.zeros((out_size, n_contrib), dtype=int)
    weights = np.zeros((out_size, n_contrib), dtype=np.float64)

    for j in range(n_contrib):
        idx = left + j
        # Distance from output point to this input pixel center
        dist = u - idx.astype(np.float64)
        weights[:, j] = _keys_cubic(dist)
        # Mirror out-of-bounds indices (matching MATLAB symmetric padding):
        # aux = [1:N, N:-1:1]; indices = aux(mod(indices-1, 2*N) + 1)
        period = 2 * in_size
        idx_mod = (idx - 1) % period  # 1-based to 0-based, then wrap
        idx_0 = np.where(idx_mod < in_size, idx_mod, period - 1 - idx_mod)
        indices[:, j] = idx_0

    # Normalize weights
    w_sum = weights.sum(axis=1, keepdims=True)
    w_sum[w_sum == 0] = 1.0
    weights = weights / w_sum

    # Apply interpolation
    if axis == 0:
        # Interpolating along rows (height)
        result = np.zeros((out_size, img.shape[1]), dtype=np.float64)
        for j in range(n_contrib):
            result += weights[:, j:j + 1] * img[indices[:, j], :]
    else:
        # Interpolating along columns (width)
        result = np.zeros((img.shape[0], out_size), dtype=np.float64)
        for j in range(n_contrib):
            result += weights[np.newaxis, :, j] * img[:, indices[:, j]]

    return result
