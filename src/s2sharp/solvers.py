"""Conjugate gradient solver for the Z-step subproblem."""

import numpy as np

from .convolution import conv_cm
from .utils import conv2mat


def grad_cost_z(
    Z: np.ndarray,
    F: np.ndarray,
    Y: np.ndarray,
    UBTMTy: np.ndarray,
    FBM: np.ndarray,
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
        Weight matrix, shape (1, n).

    Returns
    -------
    tuple[float, np.ndarray, np.ndarray]
        J : cost value
        gradJ : gradient, shape (r, n)
        AtAg : data term contribution, shape (r, n)
    """
    n = nl * nc
    X = F @ Z
    BX = conv_cm(X, FBM, nl)
    HtHBX = Mask * BX

    ZH = conv_cm(Z, FDHC, nl)
    Zv = conv_cm(Z, FDVC, nl)
    ZHW = ZH * W
    ZVW = Zv * W

    grad_pen = conv_cm(ZHW, FDH, nl) + conv_cm(ZVW, FDV, nl)

    # q is (r,), broadcast with (r, n)
    AtAg = F.T @ conv_cm(HtHBX, np.conj(FBM), nl) + 2 * tau * (q[:, np.newaxis] * np.ones((1, n))) * grad_pen
    gradJ = AtAg - UBTMTy
    J = 0.5 * np.sum(Z * AtAg) - np.sum(Z * UBTMTy)

    return J, gradJ, AtAg


def conjugate_gradient(
    Z: np.ndarray,
    F: np.ndarray,
    Y: np.ndarray,
    UBTMTy: np.ndarray,
    FBM: np.ndarray,
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
) -> np.ndarray:
    """Conjugate gradient solver for the Z-step.

    Replicates MATLAB CG function (S2sharp.m lines 501-529).

    Parameters
    ----------
    Z : np.ndarray
        Initial Z, shape (r, n).
    max_iter : int
        Maximum CG iterations.
    tol_grad_norm : float
        Gradient norm tolerance.

    Returns
    -------
    np.ndarray
        Optimized Z, shape (r, n).
    """
    cost, grad, _ = grad_cost_z(
        Z, F, Y, UBTMTy, FBM, Mask, nl, nc, r, tau, q,
        FDH, FDV, FDHC, FDVC, W
    )
    gradnorm = np.linalg.norm(grad.ravel())

    res = -grad
    desc_dir = None

    for iteration in range(1, max_iter + 1):
        if gradnorm <= tol_grad_norm:
            break

        if iteration == 1:
            desc_dir = res.copy()
        else:
            beta = np.dot(res.ravel(), res.ravel()) / np.dot(old_res.ravel(), old_res.ravel())
            desc_dir = res + beta * desc_dir

        _, _, AtAp = grad_cost_z(
            desc_dir, F, Y, UBTMTy, FBM, Mask, nl, nc, r, tau, q,
            FDH, FDV, FDHC, FDVC, W
        )

        alpha = np.dot(res.ravel(), res.ravel()) / np.dot(desc_dir.ravel(), AtAp.ravel())
        Z = Z + alpha * desc_dir

        old_res = res.copy()
        res = res - alpha * AtAp
        gradnorm = np.linalg.norm(res.ravel())

    return Z


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
) -> np.ndarray:
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
        Weight matrix, shape (1, n).
    tol_grad_norm : float
        CG tolerance.

    Returns
    -------
    np.ndarray
        Updated Z, shape (r, n).
    """
    r = F.shape[1]
    n = nl * nc

    UBTMTy = F.T @ conv_cm(Y, np.conj(FBM), nl)

    Z = conjugate_gradient(
        Z, F, Y, UBTMTy, FBM, Mask, nl, nc, r, tau, q,
        FDH, FDV, FDHC, FDVC, W,
        tol_grad_norm=tol_grad_norm,
    )
    return Z
