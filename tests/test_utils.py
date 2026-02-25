"""Tests for s2sharp.utils module."""

import numpy as np
import pytest

from s2sharp.utils import conv2im, conv2mat, gaussian_kernel


class TestConv2ImConv2Mat:
    """Test round-trip conversions between image and matrix formats."""

    def test_round_trip_3d(self):
        """conv2im(conv2mat(X)) should return the original 3D array."""
        nl, nc, L = 10, 8, 3
        X = np.random.randn(nl, nc, L)
        X_mat = conv2mat(X)
        X_back = conv2im(X_mat, nl, nc, L)
        np.testing.assert_allclose(X_back, X)

    def test_round_trip_mat(self):
        """conv2mat(conv2im(X)) should return the original matrix."""
        L, n = 4, 20
        nl, nc = 5, 4
        X = np.random.randn(L, n)
        X_im = conv2im(X, nl, nc, L)
        X_back = conv2mat(X_im)
        np.testing.assert_allclose(X_back, X)

    def test_conv2mat_shape(self):
        """conv2mat should return (L, nl*nc)."""
        nl, nc, L = 6, 8, 5
        X = np.random.randn(nl, nc, L)
        result = conv2mat(X)
        assert result.shape == (L, nl * nc)

    def test_conv2im_shape(self):
        """conv2im should return (nl, nc, L)."""
        L, n = 3, 30
        nl, nc = 5, 6
        X = np.random.randn(L, n)
        result = conv2im(X, nl, nc, L)
        assert result.shape == (nl, nc, L)

    def test_2d_image(self):
        """conv2mat should handle 2D images as single band."""
        nl, nc = 4, 5
        X = np.random.randn(nl, nc)
        result = conv2mat(X)
        assert result.shape == (1, nl * nc)

    def test_conv2im_auto_nc(self):
        """conv2im should infer nc from array shape."""
        L, nl, nc = 3, 5, 6
        n = nl * nc
        X = np.random.randn(L, n)
        result = conv2im(X, nl)
        assert result.shape == (nl, nc, L)


class TestGaussianKernel:
    """Test Gaussian kernel generation."""

    def test_shape(self):
        """Kernel should have the requested shape."""
        k = gaussian_kernel(5, 7, 1.0)
        assert k.shape == (7, 5)

    def test_normalization(self):
        """Kernel should sum to 1."""
        k = gaussian_kernel(11, 11, 2.0)
        np.testing.assert_allclose(k.sum(), 1.0, atol=1e-10)

    def test_symmetry(self):
        """Square kernel with equal dimensions should be symmetric."""
        k = gaussian_kernel(9, 9, 1.5)
        np.testing.assert_allclose(k, k.T, atol=1e-10)

    def test_center_is_max(self):
        """Center of kernel should be the maximum value."""
        k = gaussian_kernel(7, 7, 1.0)
        assert k[3, 3] == k.max()

    def test_sigma_effect(self):
        """Larger sigma should produce a wider (flatter) kernel."""
        k_narrow = gaussian_kernel(11, 11, 0.5)
        k_wide = gaussian_kernel(11, 11, 3.0)
        # Wider kernel has smaller center value
        assert k_wide[5, 5] < k_narrow[5, 5]
