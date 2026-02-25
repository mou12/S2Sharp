"""Tests for s2sharp.optimization module."""

import numpy as np
import pytest

from s2sharp.optimization import f_step, _cost_f, _egrad_f


class TestCostAndGradient:
    """Test cost function and gradient computation."""

    def test_cost_nonnegative(self):
        """Cost should be non-negative (sum of squared norms)."""
        L, r, n = 5, 3, 20
        F = np.random.randn(L, r)
        MBZT_T = np.random.randn(L, n, r)  # Transposed layout (L, n, r)
        Y = np.random.randn(L, n)
        cost = _cost_f(F, MBZT_T, Y)
        assert cost >= 0

    def test_gradient_shape(self):
        """Gradient should have same shape as F."""
        L, r = 5, 3
        F = np.random.randn(L, r)
        A = np.random.randn(r, r, L)
        ZBYT = np.random.randn(L, r)
        grad = _egrad_f(F, A, ZBYT)
        assert grad.shape == F.shape


class TestFStep:
    """Test the F-step optimization on Stiefel manifold."""

    def test_output_on_stiefel(self):
        """F-step output should satisfy F.T @ F = I (Stiefel manifold)."""
        L, r = 6, 3
        nl, nc = 10, 10
        n = nl * nc

        # Create a valid initial F on Stiefel manifold
        F0, _ = np.linalg.qr(np.random.randn(L, r))
        Z = np.random.randn(r, n) * 0.1
        Y = np.random.randn(L, n) * 0.1
        FBM = np.ones((nl, nc, L), dtype=complex)
        Mask = np.ones((L, n))

        F1 = f_step(F0, Z, Y, FBM, nl, nc, Mask)

        # Check orthonormality: F.T @ F should be close to identity
        np.testing.assert_allclose(F1.T @ F1, np.eye(r), atol=1e-6)

    def test_output_shape(self):
        """F-step should return same shape as input F."""
        L, r = 6, 3
        nl, nc = 10, 10
        n = nl * nc

        F0, _ = np.linalg.qr(np.random.randn(L, r))
        Z = np.random.randn(r, n) * 0.1
        Y = np.random.randn(L, n) * 0.1
        FBM = np.ones((nl, nc, L), dtype=complex)
        Mask = np.ones((L, n))

        F1 = f_step(F0, Z, Y, FBM, nl, nc, Mask)
        assert F1.shape == (L, r)
