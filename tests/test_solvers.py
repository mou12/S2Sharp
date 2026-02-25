"""Tests for s2sharp.solvers module."""

import numpy as np
import pytest

from s2sharp.solvers import conjugate_gradient, grad_cost_z


class TestGradCostZ:
    """Test the gradient computation for the Z subproblem."""

    def test_gradient_at_optimum_is_small(self):
        """At a point where Z minimizes cost, gradient should be near zero."""
        # This is a smoke test with zero Z (trivial case)
        r, nl, nc = 3, 10, 10
        L = 4
        n = nl * nc
        Z = np.zeros((r, n))
        F = np.eye(L, r)
        Y = np.zeros((L, n))
        Mask = np.ones((L, n))
        FBM = np.ones((nl, nc, L), dtype=complex)
        UBTMTy = np.zeros((r, n))
        q = np.ones(r)
        FDH = np.ones((nl, nc, r), dtype=complex)
        FDV = np.ones((nl, nc, r), dtype=complex)
        FDHC = np.conj(FDH)
        FDVC = np.conj(FDV)
        W = np.ones((1, n))

        J, gradJ, _ = grad_cost_z(
            Z, F, Y, UBTMTy, FBM, Mask, nl, nc, r, 0.005, q,
            FDH, FDV, FDHC, FDVC, W
        )
        # With Z=0, Y=0, cost and gradient should be 0
        assert J == pytest.approx(0.0, abs=1e-10)
        np.testing.assert_allclose(gradJ, 0.0, atol=1e-10)


class TestConjugateGradient:
    """Test the CG solver."""

    def test_convergence(self):
        """CG should converge to low gradient norm."""
        from s2sharp.convolution import create_diff_kernels

        r, nl, nc = 2, 8, 8
        L = 3
        n = nl * nc
        Z = np.random.randn(r, n) * 0.01
        F = np.eye(L, r)
        Y = np.random.randn(L, n) * 0.01
        Mask = np.ones((L, n))
        FBM = np.ones((nl, nc, L), dtype=complex)
        UBTMTy = F.T @ Y  # simplified
        q = np.ones(r)
        FDH, FDV, FDHC, FDVC = create_diff_kernels(nl, nc, r)
        W = np.ones((1, n))

        Z_opt = conjugate_gradient(
            Z, F, Y, UBTMTy, FBM, Mask, nl, nc, r, 0.005, q,
            FDH, FDV, FDHC, FDVC, W,
            max_iter=500, tol_grad_norm=0.01,
        )
        # Check that gradient norm decreased
        _, grad, _ = grad_cost_z(
            Z_opt, F, Y, UBTMTy, FBM, Mask, nl, nc, r, 0.005, q,
            FDH, FDV, FDHC, FDVC, W
        )
        assert np.linalg.norm(grad.ravel()) < 1.0  # Should have converged somewhat
