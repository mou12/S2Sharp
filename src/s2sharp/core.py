"""Main S2Sharp algorithm: Sentinel-2 sharpening using a reduced-rank method."""

import time
from dataclasses import dataclass, field

import numpy as np

from .constants import (
    DEFAULT_CDITER,
    DEFAULT_KERNEL_SIZE,
    DEFAULT_LAMBDA,
    DEFAULT_Q_R7,
    DEFAULT_RANK,
    DOWNSAMPLE_FACTORS,
    MTF,
    NUM_BANDS,
    DEFAULT_BORDER,
)
from .convolution import conv_cm, create_conv_kernel, create_diff_kernels
from .initialization import initialize
from .metrics import evaluate
from .optimization import f_step
from .preprocessing import compute_weights, normalize_data, normalize_data_image
from .solvers import z_step
from .utils import conv2im, conv2mat


@dataclass
class S2SharpResult:
    """Result container for the S2Sharp algorithm."""

    image: np.ndarray
    SAMm: list[float] = field(default_factory=list)
    SAMm_2m: list[float] = field(default_factory=list)
    SRE: list[np.ndarray] = field(default_factory=list)
    RMSE: list[float] = field(default_factory=list)
    SSIM: list[np.ndarray] = field(default_factory=list)
    aSSIM: list[float] = field(default_factory=list)
    ERGAS_20m: list[float] = field(default_factory=list)
    ERGAS_60m: list[float] = field(default_factory=list)
    GCVscore: float | None = None
    Time: float = 0.0


def s2sharp(
    bands: list[np.ndarray],
    *,
    cd_iter: int = DEFAULT_CDITER,
    r: int = DEFAULT_RANK,
    lam: float = DEFAULT_LAMBDA,
    q: np.ndarray | None = None,
    ground_truth: np.ndarray | None = None,
    x0: np.ndarray | None = None,
    g_step_only: bool = False,
    gcv: bool = False,
    tol_grad_norm: float = 0.1,
) -> S2SharpResult:
    """Main S2Sharp algorithm.

    Replicates MATLAB S2sharp() function (S2sharp.m lines 1-182).

    Sharpens Sentinel-2 bands (60m bands B1,B9 and 20m bands B5,B6,B7,B8A,B11,B12)
    to 10m resolution using a reduced-rank method.

    Parameters
    ----------
    bands : list[np.ndarray]
        List of 12 band images (possibly different sizes).
    cd_iter : int
        Number of cyclic descent iterations.
    r : int
        Subspace dimension (rank).
    lam : float
        Regularization parameter.
    q : np.ndarray, optional
        Penalty weights of length r. Defaults to [1, 1.5, 4, 8, 15, 15, 20] for r=7.
    ground_truth : np.ndarray, optional
        Ground truth image (nl, nc, L) for evaluation.
    x0 : np.ndarray, optional
        Initial value for X = G * F'.
    g_step_only : bool
        If True, only perform the G-step (once).
    gcv : bool
        If True, compute GCV score.
    tol_grad_norm : float
        Gradient norm tolerance for CG solver.

    Returns
    -------
    S2SharpResult
        Result containing sharpened image and quality metrics.
    """
    t_start = time.perf_counter()

    # Set default q
    if q is None:
        if r == DEFAULT_RANK:
            q = DEFAULT_Q_R7.copy()
        else:
            q = np.ones(r)
    q = np.asarray(q, dtype=np.float64).ravel()

    if len(q) != r:
        raise ValueError(f"Length of q ({len(q)}) must match r ({r})")

    # Dimensions
    L = len(bands)
    bands = [b.astype(np.float64) for b in bands]
    # Reference dimensions from band 2 (B2, index 1) which is at 10m
    nl, nc = bands[1].shape
    n = nl * nc

    # Normalize data
    bands_norm, av = normalize_data(bands)

    # Sentinel-2 parameters
    d = DOWNSAMPLE_FACTORS.copy()
    mtf = MTF.copy()
    sdf = d * np.sqrt(-2 * np.log(mtf) / np.pi ** 2)
    sdf[d == 1] = 0

    limsub = DEFAULT_BORDER
    dx, dy = DEFAULT_KERNEL_SIZE

    # Build convolution kernels
    FBM = create_conv_kernel(sdf, d, nl, nc, L, dx, dy)

    # Initialize
    Y, M, F = initialize(bands_norm, sdf, nl, nc, L, dx, dy, d, limsub, r)
    Mask = M.reshape(n, L).T  # (L, n)

    # Initialize Z
    if x0 is not None:
        X0_norm, _ = normalize_data_image(x0)
        X0_mat = X0_norm.reshape(n, L).T  # (L, n)
        F_svd, D_svd, Vt_svd = np.linalg.svd(X0_mat, full_matrices=False)
        F = F_svd[:, :r]
        Z = np.diag(D_svd[:r]) @ Vt_svd[:r, :]
    else:
        Z = np.zeros((r, n))

    # Difference kernels
    FDH, FDV, FDHC, FDVC = create_diff_kernels(nl, nc, r)

    # Compute weights
    sigmas = 1.0
    W = compute_weights(Y, d, sigmas, nl)

    # Output structure
    output = S2SharpResult(image=np.zeros((nl, nc, L)))

    if gcv:
        g_step_only = True
    if g_step_only:
        cd_iter = 1

    for j in range(cd_iter):
        # Z-step
        Z = z_step(
            Y, FBM, F, lam, nl, nc, Z, Mask, q,
            FDH, FDV, FDHC, FDVC, W,
            tol_grad_norm=tol_grad_norm,
        )

        # F-step (skip if g_step_only)
        if not g_step_only:
            F = f_step(F, Z, Y, FBM, nl, nc, Mask)

        # GCV computation
        if gcv:
            rng = np.random.default_rng()
            Ynoise = (np.abs(Y) > 0).astype(np.float64) * rng.standard_normal(Y.shape)
            Znoise = z_step(
                Ynoise, FBM, F, lam, nl, nc, Z, Mask, q,
                FDH, FDV, FDHC, FDVC, W,
                tol_grad_norm=tol_grad_norm,
            )
            HtHBXnoise = Mask * conv_cm(F @ Znoise, FBM, nl)

            # Indices for non-10m bands: B1(0), B5(4), B6(5), B7(6), B8A(8), B9(9), B11(10), B12(11)
            gcv_bands = [0, 4, 5, 6, 8, 9, 10, 11]
            Ynoise_sub = Ynoise[gcv_bands, :]
            HtHBXnoise_sub = HtHBXnoise[gcv_bands, :]
            den = np.trace(Ynoise_sub @ (Ynoise_sub - HtHBXnoise_sub).T)

            HtHBX = Mask * conv_cm(F @ Z, FBM, nl)
            Y_sub = Y[gcv_bands, :]
            HtHBX_sub = HtHBX[gcv_bands, :]
            num = np.linalg.norm(Y_sub - HtHBX_sub, 'fro') ** 2

            output.GCVscore = float(num / den)

        output.Time = time.perf_counter() - t_start

        # Evaluate if ground truth is available
        if ground_truth is not None:
            Xhat_im = conv2im(F @ Z, nl, nc, L)
            metrics = evaluate(ground_truth, Xhat_im, nl, nc, L, limsub, d, av)
            output.SAMm.append(metrics['SAMm'])
            output.SAMm_2m.append(metrics['SAMm_2m'])
            output.SRE.append(metrics['SRE'])
            output.RMSE.append(metrics['RMSE'])
            output.SSIM.append(metrics['SSIM'])
            output.aSSIM.append(metrics['aSSIM'])
            output.ERGAS_20m.append(metrics['ERGAS_20m'])
            output.ERGAS_60m.append(metrics['ERGAS_60m'])

    # Final image
    Xhat_im = conv2im(F @ Z, nl, nc, L)
    Xhat_im = Xhat_im[limsub:-limsub, limsub:-limsub, :]
    from .preprocessing import unnormalize_data
    Xhat_im = unnormalize_data(Xhat_im, av)

    output.image = Xhat_im
    output.Time = time.perf_counter() - t_start

    return output
