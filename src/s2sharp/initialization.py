"""SVD-based subspace initialization."""

import numpy as np

from .convolution import create_conv_kernel_subspace
from .preprocessing import create_subsampling
from .utils import conv2mat, matlab_imresize


def initialize(
    bands: list[np.ndarray],
    sdf: np.ndarray,
    nl: int,
    nc: int,
    L: int,
    dx: int,
    dy: int,
    d: np.ndarray,
    border: int,
    r: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SVD-based subspace initialization.

    Replicates MATLAB initialization function (S2sharp.m lines 184-195).

    Steps:
    1. Upsample all bands to 10m via imresize (scipy.ndimage.zoom)
    2. Apply complementary blur via FFT to equalize blur across bands
    3. Crop borders
    4. Compute SVD, take first r columns of U as initial F
    5. Create subsampling masks

    Parameters
    ----------
    bands : list[np.ndarray]
        Normalized band images.
    sdf : np.ndarray
        Standard deviations for blur kernels.
    nl, nc : int
        Image dimensions at 10m resolution.
    L : int
        Number of bands.
    dx, dy : int
        Kernel support size.
    d : np.ndarray
        Downsampling factors.
    border : int
        Border width to crop.
    r : int
        Subspace rank.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Y : observed data matrix, shape (L, nl*nc)
        M : subsampling mask matrix, shape (nl, nc, L)
        F : initial subspace matrix, shape (L, r)
    """
    # Build complementary blur kernels
    FBM2 = create_conv_kernel_subspace(sdf, nl, nc, L, dx, dy)

    # Upsample all bands to 10m resolution
    Ylim = np.zeros((nl, nc, L))
    for i in range(L):
        di = d[i]
        if di == 1:
            Ylim[:, :, i] = bands[i]
        else:
            # imresize with factor d[i] — matching MATLAB's Keys cubic kernel
            Ylim[:, :, i] = matlab_imresize(bands[i], di)

    # Apply complementary blur
    Y2im = np.real(np.fft.ifft2(np.fft.fft2(Ylim, axes=(0, 1)) * FBM2, axes=(0, 1)))

    # Crop borders
    Y2tr = Y2im[border:-border, border:-border, :]

    # Reshape and compute SVD
    nl_crop = nl - 2 * border
    nc_crop = nc - 2 * border
    Y2n = Y2tr.reshape(nl_crop * nc_crop, L)
    # SVD: Y2n.T = F * D * P' (MATLAB convention)
    F_full, _, _ = np.linalg.svd(Y2n.T, full_matrices=False)
    F = F_full[:, :r]

    # Create subsampling masks and observed data
    M, Y = create_subsampling(bands, d, nl, nc, L)

    return Y, M, F
