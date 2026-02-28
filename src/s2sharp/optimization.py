"""F-step: pymanopt Stiefel manifold optimization."""

import warnings

import numpy as np
import pymanopt
from pymanopt.manifolds import Stiefel
from pymanopt.optimizers import TrustRegions

from .convolution import conv_cm


def _compute_mbzt(
    Z: np.ndarray,
    Mask: np.ndarray,
    FBM: np.ndarray,
    nl: int,
    r: int,
    L: int,
) -> np.ndarray:
    """Precompute MBZT matrices for each band.

    MBZT(:,:,i) = repmat(Mask(i,:), [r,1]) .* ConvCM(Z, repmat(FBM(:,:,i), [1,1,r]), nl)

    Returns MBZT_T in transposed layout (L, n, r) for efficient batched operations.
    """
    n = Z.shape[1]
    MBZT_T = np.zeros((L, n, r))
    for i in range(L):
        FBM_rep = np.tile(FBM[:, :, i:i + 1], (1, 1, r))
        BZ = conv_cm(Z, FBM_rep, nl)
        # BZ is (r, n), mask is (1, n), result is (r, n)
        masked = np.tile(Mask[i:i + 1, :], (r, 1)) * BZ
        MBZT_T[i, :, :] = masked.T  # (n, r)
    return MBZT_T


def _cost_f(F: np.ndarray, MBZT_T: np.ndarray, Y: np.ndarray) -> float:
    """Cost function for the F-step on the Stiefel manifold.

    Vectorized: uses batched matmul (L, n, r) @ (L, r, 1) -> (L, n, 1).
    """
    pred = (MBZT_T @ F[:, :, np.newaxis]).squeeze(-1)  # (L, n)
    residual = pred - Y
    return 0.5 * float(np.sum(residual ** 2))


def _egrad_f(F: np.ndarray, A: np.ndarray, ZBYT: np.ndarray) -> np.ndarray:
    """Euclidean gradient of the F-step cost.

    Du(i,:) = F(i,:) * A(:,:,i)' - ZBYT(i,:)
    """
    L = A.shape[2]
    Du = np.zeros_like(F)
    for i in range(L):
        Du[i, :] = F[i, :] @ A[:, :, i].T - ZBYT[i, :]
    return Du


def f_step(
    F: np.ndarray,
    Z: np.ndarray,
    Y: np.ndarray,
    FBM: np.ndarray,
    nl: int,
    nc: int,
    Mask: np.ndarray,
) -> np.ndarray:
    """Optimize F on the Stiefel manifold using pymanopt TrustRegions.

    Replicates MATLAB Fstep (S2sharp.m lines 207-226).
    The cost J(F) = 0.5 * sum_i ||MBZT_i' * f_i - y_i||^2 is quadratic in F,
    so the Euclidean Hessian-vector product is exact.

    Parameters
    ----------
    F : np.ndarray
        Current subspace matrix, shape (L, r).
    Z : np.ndarray
        Current coefficients, shape (r, n).
    Y : np.ndarray
        Observed data, shape (L, n).
    FBM : np.ndarray
        FFT of blur kernels, shape (nl, nc, L).
    nl, nc : int
        Image dimensions.
    Mask : np.ndarray
        Subsampling mask, shape (L, n).

    Returns
    -------
    np.ndarray
        Optimized F, shape (L, r).
    """
    L, r = F.shape

    # Precompute MBZT in transposed layout (L, n, r)
    MBZT_T = _compute_mbzt(Z, Mask, FBM, nl, r, L)

    # Precompute A and ZBYT for gradient and Hessian
    A = np.zeros((r, r, L))
    ZBYT = np.zeros((L, r))
    for i in range(L):
        mbzt_i = MBZT_T[i, :, :].T  # (r, n)
        A[:, :, i] = mbzt_i @ mbzt_i.T
        ZBYT[i, :] = mbzt_i @ Y[i, :]

    manifold = Stiefel(L, r)

    @pymanopt.function.numpy(manifold)
    def cost(F):
        return _cost_f(F, MBZT_T, Y)

    @pymanopt.function.numpy(manifold)
    def euclidean_gradient(F):
        return _egrad_f(F, A, ZBYT)

    @pymanopt.function.numpy(manifold)
    def riemannian_hessian(point, tangent_vector):
        """FD Riemannian Hessian matching MATLAB manopt's approxhessianFD."""
        norm_v = manifold.norm(point, tangent_vector)
        if norm_v < 1e-30:
            return manifold.zero_vector(point)
        # Match MATLAB manopt's getHessianFD: epsilon = 2^-14,
        # c = epsilon / norm_d, step length = epsilon ≈ 6.1e-5.
        epsilon = 2.0 ** (-14)
        c = norm_v / epsilon
        y = manifold.retraction(point, tangent_vector / c)
        rgrad_x = manifold.euclidean_to_riemannian_gradient(
            point, _egrad_f(point, A, ZBYT))
        rgrad_y = manifold.euclidean_to_riemannian_gradient(
            y, _egrad_f(y, A, ZBYT))
        transported = manifold.transport(y, point, rgrad_y)
        return c * (transported - rgrad_x)

    problem = pymanopt.Problem(
        manifold=manifold,
        cost=cost,
        euclidean_gradient=euclidean_gradient,
        riemannian_hessian=riemannian_hessian,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        optimizer = TrustRegions(
            min_gradient_norm=1e-2,
            verbosity=0,
        )
        result = optimizer.run(problem, initial_point=F)

    return result.point
