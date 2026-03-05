"""Tests for s2sharp.solvers module."""

import numpy as np
import pytest

from s2sharp.solvers import (
    _apply_hessian,
    _apply_hessian_im,
    apply_preconditioner,
    build_preconditioner,
    conjugate_gradient,
    grad_cost_z,
)


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
        W = np.ones((nl, nc, 1))

        FBM_conj = np.conj(FBM)

        J, gradJ, _ = grad_cost_z(
            Z, F, Y, UBTMTy, FBM, FBM_conj, Mask, nl, nc, r, 0.005, q,
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
        W = np.ones((nl, nc, 1))

        FBM_conj = np.conj(FBM)

        Z_opt, cg_iters = conjugate_gradient(
            Z, F, Y, UBTMTy, FBM, FBM_conj, Mask, nl, nc, r, 0.005, q,
            FDH, FDV, FDHC, FDVC, W,
            max_iter=500, tol_grad_norm=0.01,
        )
        assert cg_iters >= 0
        # Check that gradient norm decreased
        _, grad, _ = grad_cost_z(
            Z_opt, F, Y, UBTMTy, FBM, FBM_conj, Mask, nl, nc, r, 0.005, q,
            FDH, FDV, FDHC, FDVC, W
        )
        assert np.linalg.norm(grad.ravel()) < 1.0  # Should have converged somewhat


class TestBuildPreconditioner:
    """Test the Fourier-domain preconditioner construction."""

    def test_shape(self):
        """M_inv should have shape (nl, nc, r, r)."""
        r, nl, nc = 3, 10, 10
        L = 4
        F = np.eye(L, r)
        FBM = np.ones((nl, nc, L), dtype=complex)
        from s2sharp.convolution import create_diff_kernels
        FDH, FDV, _, _ = create_diff_kernels(nl, nc, r)
        q = np.ones(r)

        M_inv = build_preconditioner(F, FBM, nl, nc, r, 0.005, q, FDH, FDV)
        assert M_inv.shape == (nl, nc, r, r)

    def test_symmetric(self):
        """M_inv should be symmetric at each frequency."""
        r, nl, nc = 3, 8, 8
        L = 5
        F = np.random.randn(L, r)
        FBM = np.random.randn(nl, nc, L) + 1j * np.random.randn(nl, nc, L)
        from s2sharp.convolution import create_diff_kernels
        FDH, FDV, _, _ = create_diff_kernels(nl, nc, r)
        q = np.array([1.0, 2.0, 3.0])

        M_inv = build_preconditioner(F, FBM, nl, nc, r, 0.01, q, FDH, FDV)
        # Check symmetry: M_inv[i,j] should equal its transpose
        M_inv_T = np.transpose(M_inv, (0, 1, 3, 2))
        np.testing.assert_allclose(M_inv, M_inv_T, atol=1e-10)

    def test_spd(self):
        """M_inv should be SPD at each frequency (positive eigenvalues)."""
        r, nl, nc = 2, 4, 4
        L = 3
        F = np.random.randn(L, r)
        FBM = np.random.randn(nl, nc, L) + 1j * np.random.randn(nl, nc, L)
        from s2sharp.convolution import create_diff_kernels
        FDH, FDV, _, _ = create_diff_kernels(nl, nc, r)
        q = np.ones(r)

        M_inv = build_preconditioner(F, FBM, nl, nc, r, 0.01, q, FDH, FDV)
        for i in range(nl):
            for j in range(nc):
                eigvals = np.linalg.eigvalsh(M_inv[i, j])
                assert np.all(eigvals > 0), f"Non-positive eigenvalue at ({i},{j})"


class TestApplyPreconditioner:
    """Test the preconditioner application."""

    def test_identity_preconditioner(self):
        """Identity preconditioner should return input unchanged."""
        r, nl, nc = 3, 8, 8
        n = nl * nc
        Z = np.random.randn(r, n)
        M_inv = np.tile(np.eye(r)[None, None, :, :], (nl, nc, 1, 1))

        result = apply_preconditioner(Z, M_inv, nl, nc, r)
        np.testing.assert_allclose(result, Z, atol=1e-10)

    def test_output_shape(self):
        """Output should have same shape as input."""
        r, nl, nc = 2, 6, 6
        n = nl * nc
        Z = np.random.randn(r, n)
        M_inv = np.tile(np.eye(r)[None, None, :, :], (nl, nc, 1, 1))

        result = apply_preconditioner(Z, M_inv, nl, nc, r)
        assert result.shape == (r, n)


class TestPreconditionedCG:
    """Test that PCG converges to the same solution as CG."""

    def test_pcg_matches_cg_solution(self):
        """PCG and CG should converge to the same minimizer."""
        from s2sharp.convolution import create_diff_kernels

        rng = np.random.default_rng(42)
        r, nl, nc = 2, 8, 8
        L = 3
        n = nl * nc
        Z0 = rng.standard_normal((r, n)) * 0.01
        F = np.eye(L, r)
        Y = rng.standard_normal((L, n)) * 0.01
        Mask = np.ones((L, n))
        FBM = np.ones((nl, nc, L), dtype=complex)
        UBTMTy = F.T @ Y
        q = np.ones(r)
        tau = 0.005
        FDH, FDV, FDHC, FDVC = create_diff_kernels(nl, nc, r)
        W = np.ones((nl, nc, 1))
        tol = 1e-6

        FBM_conj = np.conj(FBM)

        # Standard CG
        Z_cg, iters_cg = conjugate_gradient(
            Z0.copy(), F, Y, UBTMTy, FBM, FBM_conj, Mask, nl, nc, r, tau, q,
            FDH, FDV, FDHC, FDVC, W,
            max_iter=500, tol_grad_norm=tol,
        )

        # PCG
        M_inv = build_preconditioner(F, FBM, nl, nc, r, tau, q, FDH, FDV)
        Z_pcg, iters_pcg = conjugate_gradient(
            Z0.copy(), F, Y, UBTMTy, FBM, FBM_conj, Mask, nl, nc, r, tau, q,
            FDH, FDV, FDHC, FDVC, W,
            max_iter=500, tol_grad_norm=tol,
            M_inv=M_inv,
        )

        # Both should reach the same solution
        np.testing.assert_allclose(Z_pcg, Z_cg, atol=1e-4)
        # PCG should use fewer or equal iterations
        assert iters_pcg <= iters_cg


class TestApplyHessianIm:
    """Test that _apply_hessian_im matches conv_cm-based reference."""

    def test_matches_conv_cm_reference(self):
        """_apply_hessian_im with rfft2 should match conv_cm-based computation."""
        from s2sharp.convolution import conv_cm, create_diff_kernels

        rng = np.random.default_rng(123)
        r, nl, nc = 3, 16, 16
        L = 4
        n = nl * nc
        nc_h = nc // 2 + 1

        Z = rng.standard_normal((r, n))
        F = rng.standard_normal((L, r))
        Mask = np.ones((L, n))
        # FBM must be FFT of real kernels for rfft2 equivalence
        real_kernels = rng.standard_normal((nl, nc, L))
        FBM = np.fft.fft2(real_kernels, axes=(0, 1))
        FBM_conj = np.conj(FBM)
        q = np.ones(r)
        tau = 0.005
        FDH, FDV, FDHC, FDVC = create_diff_kernels(nl, nc, r)
        W = rng.uniform(0.5, 1.5, (nl, nc, 1))
        W_mat = W.reshape(n, 1).T  # (1, n)

        # Reference using conv_cm (old approach)
        X = F @ Z
        BX = conv_cm(X, FBM, nl)
        BX *= Mask
        ZH = conv_cm(Z, FDHC, nl)
        Zv = conv_cm(Z, FDVC, nl)
        grad_pen = conv_cm(ZH * W_mat, FDH, nl) + conv_cm(Zv * W_mat, FDV, nl)
        ref = F.T @ conv_cm(BX, FBM_conj, nl) + 2 * tau * q[:, np.newaxis] * grad_pen

        # New implementation using rfft2
        Z_im = Z.T.reshape(nl, nc, r)
        Mask_im = Mask.T.reshape(nl, nc, L)
        result_im = _apply_hessian_im(
            Z_im, F,
            FBM[:, :nc_h, :], FBM_conj[:, :nc_h, :],
            Mask_im, nl, nc, tau, q,
            FDH[:, :nc_h, :], FDV[:, :nc_h, :],
            FDHC[:, :nc_h, :], FDVC[:, :nc_h, :], W,
        )
        result = result_im.reshape(n, r).T

        np.testing.assert_allclose(result, ref, rtol=1e-10, atol=1e-10)

    def test_odd_nc(self):
        """_apply_hessian_im should work correctly with odd nc."""
        from s2sharp.convolution import conv_cm, create_diff_kernels

        rng = np.random.default_rng(456)
        r, nl, nc = 2, 8, 9  # odd nc
        L = 3
        n = nl * nc
        nc_h = nc // 2 + 1

        Z = rng.standard_normal((r, n))
        F = rng.standard_normal((L, r))
        Mask = np.ones((L, n))
        real_kernels = rng.standard_normal((nl, nc, L))
        FBM = np.fft.fft2(real_kernels, axes=(0, 1))
        FBM_conj = np.conj(FBM)
        q = np.ones(r)
        tau = 0.01
        FDH, FDV, FDHC, FDVC = create_diff_kernels(nl, nc, r)
        W = np.ones((nl, nc, 1))

        # Reference using conv_cm
        W_mat = W.reshape(n, 1).T
        X = F @ Z
        BX = conv_cm(X, FBM, nl)
        BX *= Mask
        ZH = conv_cm(Z, FDHC, nl)
        Zv = conv_cm(Z, FDVC, nl)
        grad_pen = conv_cm(ZH * W_mat, FDH, nl) + conv_cm(Zv * W_mat, FDV, nl)
        ref = F.T @ conv_cm(BX, FBM_conj, nl) + 2 * tau * q[:, np.newaxis] * grad_pen

        # rfft2-based implementation
        Z_im = Z.T.reshape(nl, nc, r)
        Mask_im = Mask.T.reshape(nl, nc, L)
        result_im = _apply_hessian_im(
            Z_im, F,
            FBM[:, :nc_h, :], FBM_conj[:, :nc_h, :],
            Mask_im, nl, nc, tau, q,
            FDH[:, :nc_h, :], FDV[:, :nc_h, :],
            FDHC[:, :nc_h, :], FDVC[:, :nc_h, :], W,
        )
        result = result_im.reshape(n, r).T

        np.testing.assert_allclose(result, ref, rtol=1e-10, atol=1e-10)


class TestApplyHessian:
    """Test that _apply_hessian matches the AtAg output of grad_cost_z."""

    def test_matches_grad_cost_z(self):
        """_apply_hessian should return the same AtAg as grad_cost_z."""
        from s2sharp.convolution import create_diff_kernels

        rng = np.random.default_rng(42)
        r, nl, nc = 3, 10, 10
        L = 4
        n = nl * nc
        Z = rng.standard_normal((r, n))
        F = rng.standard_normal((L, r))
        Y = rng.standard_normal((L, n))
        Mask = np.ones((L, n))
        FBM = rng.standard_normal((nl, nc, L)) + 1j * rng.standard_normal((nl, nc, L))
        FBM_conj = np.conj(FBM)
        UBTMTy = F.T @ Y  # simplified
        q = np.ones(r)
        tau = 0.005
        FDH, FDV, FDHC, FDVC = create_diff_kernels(nl, nc, r)
        W = rng.uniform(0.5, 1.5, (nl, nc, 1))

        _, _, AtAg_ref = grad_cost_z(
            Z, F, Y, UBTMTy, FBM, FBM_conj, Mask, nl, nc, r, tau, q,
            FDH, FDV, FDHC, FDVC, W,
        )

        AtAg = _apply_hessian(
            Z, F, FBM, FBM_conj, Mask, nl, nc, tau, q,
            FDH, FDV, FDHC, FDVC, W,
        )

        np.testing.assert_allclose(AtAg, AtAg_ref, rtol=1e-12, atol=1e-12)
