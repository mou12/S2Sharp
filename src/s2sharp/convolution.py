"""FFT-based convolution kernels and operators for Sentinel-2 sharpening."""

import numpy as np
import scipy.fft

from .utils import conv2im, conv2mat, gaussian_kernel


def create_conv_kernel(
    sdf: np.ndarray,
    d: np.ndarray,
    nl: int,
    nc: int,
    L: int,
    dx: int,
    dy: int,
) -> np.ndarray:
    """Build FFT of Gaussian blur kernels for each band.

    Replicates MATLAB createConvKernel (S2sharp.m lines 329-353).

    Parameters
    ----------
    sdf : np.ndarray
        Standard deviation of blur for each band, shape (L,).
    d : np.ndarray
        Downsampling factors, shape (L,).
    nl, nc : int
        Image dimensions (rows, columns) at 10m resolution.
    L : int
        Number of bands.
    dx, dy : int
        Kernel support size.

    Returns
    -------
    np.ndarray
        FFT of blur kernels, shape (nl, nc, L).
    """
    middlel = nl // 2
    middlec = nc // 2

    FBM = np.zeros((nl, nc, L), dtype=complex)

    for i in range(L):
        B = np.zeros((nl, nc))
        if d[i] > 1:
            h = gaussian_kernel(dx, dy, sdf[i])
            # Place kernel centered, shifted by d/2
            # MATLAB: B(middlel-dy/2+1:middlel+dy/2 - d(i)/2+1, middlec-dx/2+1:middlec+dx/2 - d(i)/2+1, i) = h
            row_start = middlel - dy // 2 - d[i] // 2 + 1
            row_end = row_start + dy
            col_start = middlec - dx // 2 - d[i] // 2 + 1
            col_end = col_start + dx
            B[row_start:row_end, col_start:col_end] = h

            B = np.fft.fftshift(B)
            B = B / B.sum()
            FBM[:, :, i] = np.fft.fft2(B)
        else:
            B[0, 0] = 1
            FBM[:, :, i] = np.fft.fft2(B)

    return FBM


def create_conv_kernel_subspace(
    sdf: np.ndarray,
    nl: int,
    nc: int,
    L: int,
    dx: int,
    dy: int,
) -> np.ndarray:
    """Build complementary blur kernels for subspace initialization.

    Replicates MATLAB createConvKernelSubspace (S2sharp.m lines 356-390).
    Creates kernels that equalize the blur across bands by applying
    complementary Gaussian blur so all bands have the same effective blur.

    Parameters
    ----------
    sdf : np.ndarray
        Standard deviation of blur for each band, shape (L,).
    nl, nc : int
        Image dimensions at 10m resolution.
    L : int
        Number of bands.
    dx, dy : int
        Base kernel support size (will be incremented by 1).

    Returns
    -------
    np.ndarray
        FFT of complementary blur kernels, shape (nl, nc, L).
    """
    middlel = round((nl + 1) / 2)  # MATLAB: round((nl+1)/2), 1-based
    middlec = round((nc + 1) / 2)

    # Convert to 0-based
    middlel -= 1
    middlec -= 1

    dx = dx + 1
    dy = dy + 1

    FBM2 = np.zeros((nl, nc, L), dtype=complex)
    s2 = np.max(sdf)

    for i in range(L):
        B = np.zeros((nl, nc))
        if sdf[i] < s2:
            sigma_comp = np.sqrt(s2 ** 2 - sdf[i] ** 2)
            h = gaussian_kernel(dx, dy, sigma_comp)
            # Place kernel centered at middlel, middlec
            # MATLAB: B(middlel-(dy-1)/2:middlel+(dy-1)/2, middlec-(dx-1)/2:middlec+(dx-1)/2, i) = h
            half_dy = (dy - 1) // 2
            half_dx = (dx - 1) // 2
            row_start = middlel - half_dy
            row_end = middlel + half_dy + 1
            col_start = middlec - half_dx
            col_end = middlec + half_dx + 1
            B[row_start:row_end, col_start:col_end] = h

            B = np.fft.fftshift(B)
            B = B / B.sum()
            FBM2[:, :, i] = np.fft.fft2(B)
        else:
            B[0, 0] = 1
            FBM2[:, :, i] = np.fft.fft2(B)

    return FBM2


def conv_cm(
    X: np.ndarray,
    FKM: np.ndarray,
    nl: int,
) -> np.ndarray:
    """Circular convolution via FFT: conv2mat(real(ifft2(fft2(conv2im(X)) .* FKM))).

    Replicates MATLAB ConvCM (S2sharp.m line 398).

    Parameters
    ----------
    X : np.ndarray
        Input in matrix format, shape (L, n) where n = nl * nc.
    FKM : np.ndarray
        FFT of convolution kernels, shape (nl, nc, L).
    nl : int
        Number of rows in image.

    Returns
    -------
    np.ndarray
        Convolved result in matrix format, shape (L, n).
    """
    L = X.shape[0]
    n = X.shape[1]
    nc = n // nl
    # Inline conv2im: X.T.reshape(nl, nc, L)
    X_im = X.T.reshape(nl, nc, L)
    # FFT convolution using scipy.fft for better performance
    result = np.real(scipy.fft.ifft2(scipy.fft.fft2(X_im, axes=(0, 1)) * FKM, axes=(0, 1)))
    # Inline conv2mat: result.reshape(n, L).T
    return result.reshape(n, L).T


def create_diff_kernels(nl: int, nc: int, r: int) -> tuple[np.ndarray, ...]:
    """Create horizontal and vertical finite difference kernels in FFT domain.

    Replicates MATLAB createDiffkernels (S2sharp.m lines 293-304).

    Parameters
    ----------
    nl, nc : int
        Image dimensions.
    r : int
        Subspace rank (number of components).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        FDH, FDV, FDHC, FDVC — FFT of difference kernels and their conjugates,
        each of shape (nl, nc, r).
    """
    dh = np.zeros((nl, nc))
    dh[0, 0] = 1
    dh[0, nc - 1] = -1

    dv = np.zeros((nl, nc))
    dv[0, 0] = 1
    dv[nl - 1, 0] = -1

    FDH = np.tile(np.fft.fft2(dh)[:, :, np.newaxis], (1, 1, r))
    FDV = np.tile(np.fft.fft2(dv)[:, :, np.newaxis], (1, 1, r))
    FDHC = np.conj(FDH)
    FDVC = np.conj(FDV)

    return FDH, FDV, FDHC, FDVC
