"""Conjugate gradient solver for the Z-step subproblem."""

import numpy as np
import scipy.fft

from .convolution import conv_cm


def build_preconditioner(
    F: np.ndarray,
    FBM: np.ndarray,
    nl: int,
    nc: int,
    r: int,
    tau: float,
    q: np.ndarray,
    FDH: np.ndarray,
    FDV: np.ndarray,
) -> np.ndarray:
    """Build Fourier-domain preconditioner for the Z-step CG.

    Approximates the Hessian by assuming Mask=I and W=1 (circulant
    approximation), then inverts the resulting r×r matrix at each
    frequency.

    Parameters
    ----------
    F : np.ndarray
        Subspace matrix, shape (L, r).
    FBM : np.ndarray
        FFT of blur kernels, shape (nl, nc, L).
    nl, nc : int
        Image dimensions.
    r : int
        Subspace rank.
    tau : float
        Regularization parameter.
    q : np.ndarray
        Penalty weights, shape (r,).
    FDH, FDV : np.ndarray
        FFT of difference kernels, shape (nl, nc, r).

    Returns
    -------
    np.ndarray
        Inverse preconditioner M_inv, shape (nl, nc, r, r).
    """
    FBM_sq = np.abs(FBM) ** 2  # (nl, nc, L)
    D_data = np.einsum('la,ijl,lb->ijab', F, FBM_sq, F)  # (nl, nc, r, r)

    diff_sq = np.abs(FDH[:, :, 0]) ** 2 + np.abs(FDV[:, :, 0]) ** 2  # (nl, nc)
    D_reg = np.zeros((nl, nc, r, r))
    idx = np.arange(r)
    D_reg[:, :, idx, idx] = 2 * tau * q[None, None, :] * diff_sq[:, :, None]

    M = D_data + D_reg + 1e-10 * np.eye(r)[None, None, :, :]
    return np.linalg.inv(M)


def apply_preconditioner(
    Z: np.ndarray,
    M_inv: np.ndarray,
    nl: int,
    nc: int,
    r: int,
) -> np.ndarray:
    """Apply Fourier-domain preconditioner to a vector Z.

    Parameters
    ----------
    Z : np.ndarray
        Input, shape (r, n) where n = nl*nc.
    M_inv : np.ndarray
        Inverse preconditioner, shape (nl, nc, r, r).
    nl, nc : int
        Image dimensions.
    r : int
        Subspace rank.

    Returns
    -------
    np.ndarray
        Preconditioned vector, shape (r, n).
    """
    Z_im = Z.T.reshape(nl, nc, r)
    Z_freq = scipy.fft.fft2(Z_im, axes=(0, 1), workers=-1)
    Z_precond = np.einsum('ijab,ijb->ija', M_inv, Z_freq)
    return np.real(scipy.fft.ifft2(Z_precond, axes=(0, 1), workers=-1)).reshape(nl * nc, r).T


def _apply_hessian_im(
    Z_im: np.ndarray,
    F: np.ndarray,
    FBM_h: np.ndarray,
    FBM_conj_h: np.ndarray,
    Mask_im: np.ndarray,
    nl: int,
    nc: int,
    tau: float,
    q: np.ndarray,
    FDH_h: np.ndarray,
    FDV_h: np.ndarray,
    FDHC_h: np.ndarray,
    FDVC_h: np.ndarray,
    W: np.ndarray,
) -> np.ndarray:
    """Apply the Hessian in image format using rfft2.

    Works entirely in (nl, nc, bands) image format with half-spectrum
    FFTs for real inputs. Eliminates format conversions and uses rfft2
    for ~40-50% faster FFTs on real data.

    Parameters
    ----------
    Z_im : np.ndarray
        Input in image format, shape (nl, nc, r).
    F : np.ndarray
        Subspace matrix, shape (L, r).
    FBM_h : np.ndarray
        Half-spectrum blur kernels, shape (nl, nc//2+1, L).
    FBM_conj_h : np.ndarray
        Conjugate half-spectrum blur kernels, shape (nl, nc//2+1, L).
    Mask_im : np.ndarray
        Mask in image format, shape (nl, nc, L).
    nl, nc : int
        Image dimensions.
    tau : float
        Regularization parameter.
    q : np.ndarray
        Penalty weights, shape (r,).
    FDH_h, FDV_h, FDHC_h, FDVC_h : np.ndarray
        Half-spectrum difference kernels, shape (nl, nc//2+1, r).
    W : np.ndarray
        Weight image, shape (nl, nc, 1).

    Returns
    -------
    np.ndarray
        Hessian-vector product in image format, shape (nl, nc, r).
    """
    # Data term: F@Z via einsum, blur via rfft2
    X_im = np.einsum('La,ija->ijL', F, Z_im)
    X_freq = scipy.fft.rfft2(X_im, axes=(0, 1), workers=-1)
    BX_im = scipy.fft.irfft2(X_freq * FBM_h, s=(nl, nc), axes=(0, 1), workers=-1)
    BX_im *= Mask_im
    BX_freq = scipy.fft.rfft2(BX_im, axes=(0, 1), workers=-1)
    adj_im = scipy.fft.irfft2(BX_freq * FBM_conj_h, s=(nl, nc), axes=(0, 1), workers=-1)
    data_term = np.einsum('La,ijL->ija', F, adj_im)

    # Penalty term: shared rfft2 of Z, fused final irfft2
    Z_freq = scipy.fft.rfft2(Z_im, axes=(0, 1), workers=-1)
    ZH_im = scipy.fft.irfft2(Z_freq * FDHC_h, s=(nl, nc), axes=(0, 1), workers=-1)
    Zv_im = scipy.fft.irfft2(Z_freq * FDVC_h, s=(nl, nc), axes=(0, 1), workers=-1)
    ZH_im *= W
    Zv_im *= W
    ZHW_freq = scipy.fft.rfft2(ZH_im, axes=(0, 1), workers=-1)
    ZVW_freq = scipy.fft.rfft2(Zv_im, axes=(0, 1), workers=-1)
    grad_pen_im = scipy.fft.irfft2(ZHW_freq * FDH_h + ZVW_freq * FDV_h,
                                    s=(nl, nc), axes=(0, 1), workers=-1)

    return data_term + 2 * tau * q[np.newaxis, np.newaxis, :] * grad_pen_im


def _apply_preconditioner_im(
    Z_im: np.ndarray,
    M_inv_h: np.ndarray,
    nl: int,
    nc: int,
) -> np.ndarray:
    """Apply Fourier-domain preconditioner in image format using rfft2.

    Parameters
    ----------
    Z_im : np.ndarray
        Input in image format, shape (nl, nc, r).
    M_inv_h : np.ndarray
        Half-spectrum preconditioner, shape (nl, nc//2+1, r, r).
    nl, nc : int
        Image dimensions.

    Returns
    -------
    np.ndarray
        Preconditioned result in image format, shape (nl, nc, r).
    """
    Z_freq = scipy.fft.rfft2(Z_im, axes=(0, 1), workers=-1)
    Z_precond = np.einsum('ijab,ijb->ija', M_inv_h, Z_freq)
    return scipy.fft.irfft2(Z_precond, s=(nl, nc), axes=(0, 1), workers=-1)


def _apply_hessian(
    Z: np.ndarray,
    F: np.ndarray,
    FBM: np.ndarray,
    FBM_conj: np.ndarray,
    Mask: np.ndarray,
    nl: int,
    nc: int,
    tau: float,
    q: np.ndarray,
    FDH: np.ndarray,
    FDV: np.ndarray,
    FDHC: np.ndarray,
    FDVC: np.ndarray,
    W: np.ndarray,
) -> np.ndarray:
    """Apply the Hessian (A'A + regularization) to Z.

    Delegates to _apply_hessian_im for rfft2-based computation.

    Parameters
    ----------
    Z : np.ndarray
        Input, shape (r, n).
    F : np.ndarray
        Subspace matrix, shape (L, r).
    FBM : np.ndarray
        FFT of blur kernels, shape (nl, nc, L).
    FBM_conj : np.ndarray
        Conjugate of FBM, shape (nl, nc, L).
    Mask : np.ndarray
        Subsampling mask, shape (L, n).
    nl, nc : int
        Image dimensions.
    tau : float
        Regularization parameter.
    q : np.ndarray
        Penalty weights, shape (r,).
    FDH, FDV, FDHC, FDVC : np.ndarray
        FFT of difference kernels, each shape (nl, nc, r).
    W : np.ndarray
        Weight image, shape (nl, nc, 1).

    Returns
    -------
    np.ndarray
        Hessian-vector product, shape (r, n).
    """
    r = Z.shape[0]
    n = Z.shape[1]
    nc_h = nc // 2 + 1
    L = Mask.shape[0]

    Z_im = Z.T.reshape(nl, nc, r)
    Mask_im = Mask.T.reshape(nl, nc, L)

    result_im = _apply_hessian_im(
        Z_im, F,
        FBM[:, :nc_h, :], FBM_conj[:, :nc_h, :],
        Mask_im, nl, nc, tau, q,
        FDH[:, :nc_h, :], FDV[:, :nc_h, :],
        FDHC[:, :nc_h, :], FDVC[:, :nc_h, :], W,
    )

    return result_im.reshape(n, r).T


def grad_cost_z(
    Z: np.ndarray,
    F: np.ndarray,
    Y: np.ndarray,
    UBTMTy: np.ndarray,
    FBM: np.ndarray,
    FBM_conj: np.ndarray,
    Mask: np.ndarray,
    nl: int,
    nc: int,
    r: int,
    tau: float,
    q: np.ndarray,
    FDH: np.ndarray,
    FDV: np.ndarray,
    FDHC: np.ndarray,
    FDVC: np.ndarray,
    W: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Compute cost and gradient for Z subproblem.

    Replicates MATLAB grad_cost_G (S2sharp.m lines 486-499).
    Includes data fidelity term and TV regularization.

    Parameters
    ----------
    Z : np.ndarray
        Current Z, shape (r, n).
    F : np.ndarray
        Subspace matrix, shape (L, r).
    Y : np.ndarray
        Observed data, shape (L, n).
    UBTMTy : np.ndarray
        Precomputed F' * ConvCM(Y, conj(FBM)), shape (r, n).
    FBM : np.ndarray
        FFT of blur kernels, shape (nl, nc, L).
    FBM_conj : np.ndarray
        Precomputed np.conj(FBM), shape (nl, nc, L).
    Mask : np.ndarray
        Subsampling mask, shape (L, n).
    nl, nc : int
        Image dimensions.
    r : int
        Subspace rank.
    tau : float
        Regularization parameter (lambda).
    q : np.ndarray
        Penalty weights, shape (r,).
    FDH, FDV, FDHC, FDVC : np.ndarray
        FFT of difference kernels, each shape (nl, nc, r).
    W : np.ndarray
        Weight image, shape (nl, nc, 1).

    Returns
    -------
    tuple[float, np.ndarray, np.ndarray]
        J : cost value
        gradJ : gradient, shape (r, n)
        AtAg : data term contribution, shape (r, n)
    """
    AtAg = _apply_hessian(Z, F, FBM, FBM_conj, Mask, nl, nc, tau, q, FDH, FDV, FDHC, FDVC, W)
    gradJ = AtAg - UBTMTy
    J = 0.5 * np.sum(Z * AtAg) - np.sum(Z * UBTMTy)

    return J, gradJ, AtAg


def conjugate_gradient(
    Z: np.ndarray,
    F: np.ndarray,
    Y: np.ndarray,
    UBTMTy: np.ndarray,
    FBM: np.ndarray,
    FBM_conj: np.ndarray,
    Mask: np.ndarray,
    nl: int,
    nc: int,
    r: int,
    tau: float,
    q: np.ndarray,
    FDH: np.ndarray,
    FDV: np.ndarray,
    FDHC: np.ndarray,
    FDVC: np.ndarray,
    W: np.ndarray,
    max_iter: int = 1000,
    tol_grad_norm: float = 0.1,
    M_inv: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    """Conjugate gradient solver for the Z-step.

    Replicates MATLAB CG function (S2sharp.m lines 501-529).
    When M_inv is provided, uses preconditioned CG (PCG).

    Parameters
    ----------
    Z : np.ndarray
        Initial Z, shape (r, n).
    max_iter : int
        Maximum CG iterations.
    tol_grad_norm : float
        Gradient norm tolerance.
    M_inv : np.ndarray, optional
        Fourier-domain preconditioner, shape (nl, nc, r, r).
        When None, falls back to standard CG.

    Returns
    -------
    tuple[np.ndarray, int]
        Optimized Z, shape (r, n), and number of CG iterations performed.
    """
    n = nl * nc
    nc_h = nc // 2 + 1
    L = Mask.shape[0]

    # Convert to image format once at entry
    Z_im = Z.T.reshape(nl, nc, r)
    UBTMTy_im = UBTMTy.T.reshape(nl, nc, r)
    Mask_im = Mask.T.reshape(nl, nc, L)

    # Slice FFT kernels to half-spectrum for rfft2
    FBM_h = FBM[:, :nc_h, :]
    FBM_conj_h = FBM_conj[:, :nc_h, :]
    FDH_h = FDH[:, :nc_h, :]
    FDV_h = FDV[:, :nc_h, :]
    FDHC_h = FDHC[:, :nc_h, :]
    FDVC_h = FDVC[:, :nc_h, :]

    M_inv_h = M_inv[:, :nc_h, :, :] if M_inv is not None else None

    AtAg_im = _apply_hessian_im(
        Z_im, F, FBM_h, FBM_conj_h, Mask_im, nl, nc, tau, q,
        FDH_h, FDV_h, FDHC_h, FDVC_h, W,
    )
    res_im = UBTMTy_im - AtAg_im
    gradnorm = np.linalg.norm(res_im)

    if M_inv_h is not None:
        z_vec_im = _apply_preconditioner_im(res_im, M_inv_h, nl, nc)
    else:
        z_vec_im = res_im

    desc_dir_im = None
    actual_iters = 0

    for iteration in range(1, max_iter + 1):
        if gradnorm <= tol_grad_norm:
            break

        actual_iters = iteration

        rz = np.sum(res_im * z_vec_im)

        if iteration == 1:
            desc_dir_im = z_vec_im.copy()
        else:
            beta = rz / old_rz
            desc_dir_im *= beta
            desc_dir_im += z_vec_im

        AtAp_im = _apply_hessian_im(
            desc_dir_im, F, FBM_h, FBM_conj_h, Mask_im, nl, nc, tau, q,
            FDH_h, FDV_h, FDHC_h, FDVC_h, W,
        )

        alpha = rz / np.sum(desc_dir_im * AtAp_im)
        Z_im += alpha * desc_dir_im

        res_im -= alpha * AtAp_im
        gradnorm = np.linalg.norm(res_im)

        old_rz = rz
        if M_inv_h is not None:
            z_vec_im = _apply_preconditioner_im(res_im, M_inv_h, nl, nc)
        else:
            z_vec_im = res_im

    # Convert back to matrix format
    return Z_im.reshape(n, r).T.copy(), actual_iters


def z_step(
    Y: np.ndarray,
    FBM: np.ndarray,
    F: np.ndarray,
    tau: float,
    nl: int,
    nc: int,
    Z: np.ndarray,
    Mask: np.ndarray,
    q: np.ndarray,
    FDH: np.ndarray,
    FDV: np.ndarray,
    FDHC: np.ndarray,
    FDVC: np.ndarray,
    W: np.ndarray,
    tol_grad_norm: float = 0.1,
    precondition: bool = False,
) -> tuple[np.ndarray, int]:
    """Solve the Z subproblem via conjugate gradient.

    Replicates MATLAB Zstep (S2sharp.m lines 198-205).

    Parameters
    ----------
    Y : np.ndarray
        Observed data, shape (L, n).
    FBM : np.ndarray
        FFT of blur kernels, shape (nl, nc, L).
    F : np.ndarray
        Current subspace matrix, shape (L, r).
    tau : float
        Regularization parameter (lambda).
    nl, nc : int
        Image dimensions.
    Z : np.ndarray
        Current Z estimate, shape (r, n).
    Mask : np.ndarray
        Subsampling mask, shape (L, n).
    q : np.ndarray
        Penalty weights, shape (r,).
    FDH, FDV, FDHC, FDVC : np.ndarray
        FFT of difference kernels.
    W : np.ndarray
        Weight image, shape (nl, nc, 1).
    tol_grad_norm : float
        CG tolerance.
    precondition : bool
        If True, use Fourier-domain preconditioner for CG.

    Returns
    -------
    tuple[np.ndarray, int]
        Updated Z, shape (r, n), and number of CG iterations performed.
    """
    r = F.shape[1]
    n = nl * nc

    FBM_conj = np.conj(FBM)
    UBTMTy = F.T @ conv_cm(Y, FBM_conj, nl)

    M_inv = None
    if precondition:
        M_inv = build_preconditioner(F, FBM, nl, nc, r, tau, q, FDH, FDV)

    Z, cg_iters = conjugate_gradient(
        Z, F, Y, UBTMTy, FBM, FBM_conj, Mask, nl, nc, r, tau, q,
        FDH, FDV, FDHC, FDVC, W,
        tol_grad_norm=tol_grad_norm,
        M_inv=M_inv,
    )
    return Z, cg_iters
