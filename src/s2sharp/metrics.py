"""Quality metrics: SAM, ERGAS, SRE, SSIM, RMSE."""

import numpy as np
from skimage.metrics import structural_similarity


def sam(I1: np.ndarray, I2: np.ndarray) -> tuple[float, np.ndarray]:
    """Spectral Angle Mapper (SAM).

    Replicates MATLAB SAM function (S2sharp.m).

    Parameters
    ----------
    I1 : np.ndarray
        Reference image, shape (M, N, B).
    I2 : np.ndarray
        Test image, shape (M, N, B).

    Returns
    -------
    tuple[float, np.ndarray]
        SAM index (degrees) and SAM map.
    """
    M, N = I2.shape[:2]

    prod_scal = np.sum(I1 * I2, axis=2)
    norm_orig = np.sum(I1 * I1, axis=2)
    norm_fusa = np.sum(I2 * I2, axis=2)
    prod_norm = np.sqrt(norm_orig * norm_fusa)

    # SAM map
    prod_map = prod_norm.copy()
    prod_map[prod_map == 0] = np.finfo(float).eps
    SAM_map = np.arccos(np.clip(prod_scal / prod_map, -1, 1))

    # SAM index
    prod_scal_flat = prod_scal.ravel()
    prod_norm_flat = prod_norm.ravel()

    # Remove zero-norm pixels
    nonzero = prod_norm_flat != 0
    prod_scal_flat = prod_scal_flat[nonzero]
    prod_norm_flat = prod_norm_flat[nonzero]

    angolo = np.sum(np.arccos(np.clip(prod_scal_flat / prod_norm_flat, -1, 1)))
    angolo /= prod_norm_flat.shape[0]

    SAM_index = float(np.real(angolo) * 180 / np.pi)
    return SAM_index, SAM_map


def ergas(I1: np.ndarray, I2: np.ndarray, ratio: int) -> float:
    """ERGAS (Erreur Relative Globale Adimensionnelle de Synthese).

    Replicates MATLAB ERGAS function (S2sharp.m).

    Parameters
    ----------
    I1 : np.ndarray
        Reference image, shape (M, N, B).
    I2 : np.ndarray
        Test image, shape (M, N, B).
    ratio : int
        Scale ratio.

    Returns
    -------
    float
        ERGAS index.
    """
    I1 = I1.astype(np.float64)
    I2 = I2.astype(np.float64)
    Err = I1 - I2
    num_bands = Err.shape[2]

    ergas_sum = 0.0
    for b in range(num_bands):
        band_err = Err[:, :, b]
        band_ref = I1[:, :, b]
        ergas_sum += np.mean(band_err ** 2) / (np.mean(band_ref) ** 2)

    return float((100 / ratio) * np.sqrt(ergas_sum / num_bands))


def sre(X_true: np.ndarray, X_est: np.ndarray) -> np.ndarray:
    """Signal-to-Reconstruction Error per band (dB).

    Parameters
    ----------
    X_true : np.ndarray
        True data, shape (L, n) in matrix form.
    X_est : np.ndarray
        Estimated data, shape (L, n) in matrix form.

    Returns
    -------
    np.ndarray
        SRE per band in dB, shape (L,).
    """
    L = X_true.shape[0]
    result = np.zeros(L)
    for i in range(L):
        signal_power = np.sum(X_true[i, :] ** 2)
        error_power = np.sum((X_est[i, :] - X_true[i, :]) ** 2)
        result[i] = 10 * np.log10(signal_power / error_power)
    return result


def compute_ssim(I1: np.ndarray, I2: np.ndarray) -> np.ndarray:
    """Compute per-band SSIM.

    Parameters
    ----------
    I1 : np.ndarray
        Reference image, shape (M, N, B).
    I2 : np.ndarray
        Test image, shape (M, N, B).

    Returns
    -------
    np.ndarray
        SSIM per band, shape (B,).
    """
    num_bands = I1.shape[2]
    result = np.zeros(num_bands)
    for b in range(num_bands):
        data_range = I1[:, :, b].max() - I1[:, :, b].min()
        if data_range == 0:
            data_range = 1.0
        result[b] = structural_similarity(
            I1[:, :, b], I2[:, :, b], data_range=data_range
        )
    return result


def evaluate(
    Xm_im: np.ndarray,
    Xhat_im: np.ndarray,
    nl: int,
    nc: int,
    L: int,
    border: int,
    d: np.ndarray,
    av: np.ndarray,
) -> dict:
    """Compute all quality metrics.

    Replicates MATLAB evaluate_performance (S2sharp.m).

    Parameters
    ----------
    Xm_im : np.ndarray
        Ground truth image, shape (nl, nc, L) or (nl, nc, 6) for reduced resolution.
    Xhat_im : np.ndarray
        Estimated image, shape (nl, nc, L). Will be cropped and unnormalized.
    nl, nc : int
        Full image dimensions.
    L : int
        Number of bands.
    border : int
        Border width to crop.
    d : np.ndarray
        Downsampling factors.
    av : np.ndarray
        Normalization factors.

    Returns
    -------
    dict
        Dictionary with keys: SAMm, SAMm_2m, SRE, RMSE, SSIM, aSSIM,
        ERGAS_20m, ERGAS_60m.
    """
    from .preprocessing import unnormalize_data
    from .utils import conv2mat

    # Crop borders
    Xhat_crop = Xhat_im[border:-(border), border:-(border), :]
    Xhat_crop = unnormalize_data(Xhat_crop, av)

    Xm_crop = Xm_im[border:-(border), border:-(border), :]

    if Xm_crop.shape[2] == 6:
        # Reduced resolution case
        ind = np.where(d == 2)[0]
        SAMm_val = sam(Xm_crop, Xhat_crop[:, :, ind])[0]
        SAMm_2m_val = SAMm_val

        X = conv2mat(Xm_crop)
        Xhat_mat = conv2mat(Xhat_crop)

        SRE_val = np.zeros(6)
        SSIM_val = np.zeros(6)
        for i in range(6):
            SRE_val[i] = 10 * np.log10(
                np.sum(X[i, :] ** 2) / np.sum((Xhat_mat[ind[i], :] - X[i, :]) ** 2)
            )
            data_range = Xm_crop[:, :, i].max() - Xm_crop[:, :, i].min()
            if data_range == 0:
                data_range = 1.0
            SSIM_val[i] = structural_similarity(
                Xm_crop[:, :, i], Xhat_crop[:, :, ind[i]], data_range=data_range
            )

        aSSIM_val = float(np.mean(SSIM_val))
        ERGAS_20m_val = ergas(Xm_crop, Xhat_crop[:, :, ind], 2)
        ERGAS_60m_val = float('nan')
        RMSE_val = float(np.linalg.norm(X - Xhat_mat[ind, :], 'fro') / X.shape[1])
    else:
        # Full resolution case
        ind = np.where((d == 2) | (d == 6))[0]
        SAMm_val = sam(Xm_crop[:, :, ind], Xhat_crop[:, :, ind])[0]

        ind2 = np.where(d == 2)[0]
        SAMm_2m_val = sam(Xm_crop[:, :, ind2], Xhat_crop[:, :, ind2])[0]

        X = conv2mat(Xm_crop)
        Xhat_mat = conv2mat(Xhat_crop)

        SRE_val = np.zeros(L)
        SSIM_val = np.zeros(L)
        for i in range(L):
            SRE_val[i] = 10 * np.log10(
                np.sum(X[i, :] ** 2) / np.sum((Xhat_mat[i, :] - X[i, :]) ** 2)
            )
            data_range = Xm_crop[:, :, i].max() - Xm_crop[:, :, i].min()
            if data_range == 0:
                data_range = 1.0
            SSIM_val[i] = structural_similarity(
                Xm_crop[:, :, i], Xhat_crop[:, :, i], data_range=data_range
            )

        aSSIM_val = float(np.mean(SSIM_val[ind]))
        ERGAS_20m_val = ergas(Xm_crop[:, :, ind], Xhat_crop[:, :, ind], 2)
        ERGAS_60m_val = ergas(Xm_crop[:, :, ind2], Xhat_crop[:, :, ind2], 6)
        RMSE_val = float(np.linalg.norm(X[ind, :] - Xhat_mat[ind, :], 'fro') / X.shape[1])

    return {
        'SAMm': SAMm_val,
        'SAMm_2m': SAMm_2m_val,
        'SRE': SRE_val,
        'RMSE': RMSE_val,
        'SSIM': SSIM_val,
        'aSSIM': aSSIM_val,
        'ERGAS_20m': ERGAS_20m_val,
        'ERGAS_60m': ERGAS_60m_val,
    }
